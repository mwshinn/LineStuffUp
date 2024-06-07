from transform import *
import tempfile
from pathlib import Path
import numpy as np

_FIXED_TRANSFORMS = [TranslateRotateFixed, TranslateRotateFixed, TranslateFixed, Identity, Rescale, ShearFixed]
_FIXED_TRANSFORMS_PARAMS = [dict(z=3.2, y=5, x=-24, zrotate=3.4, yrotate=10, xrotate=20), dict(z=3.2, y=0, x=-24, zrotate=3.4, yrotate=10, xrotate=25), dict(z=-10, y=.3, x=4), dict(), dict(z=2, y=4, x=3), dict(yzshear=.3, xzshear=-.2, xyshear=.1)]
FIXED_TRANSFORMS = [t(**tp) for t,tp in zip(_FIXED_TRANSFORMS, _FIXED_TRANSFORMS_PARAMS)]
_POINT_TRANSFORMS = [TranslateRotate2D, Translate, TranslateRotate]
points_pre = np.random.randn(50,3)
points_post = points_pre@rotation_matrix(4,6,2)-9
POINT_TRANSFORMS = [t(points_pre, points_post) for t in _POINT_TRANSFORMS]

ALL_TRANSFORMS = FIXED_TRANSFORMS + POINT_TRANSFORMS

close = lambda x,y : np.allclose(x, y, atol=1e-3, rtol=1e-3)

# Testing all transforms
new_points = np.random.randn(100,3).astype("float32")
for t in ALL_TRANSFORMS:
    assert close(t.inverse_transform(t.transform(new_points)), new_points), f"Transform {t} can not be inverted"

# Invert all transforms
for t in ALL_TRANSFORMS:
    assert close(t.transform(t.invert().transform(new_points)), new_points), f"Error with inverse of transform {t}"

# Test all compositions for inversion identity
for t1 in ALL_TRANSFORMS:
    for t2 in ALL_TRANSFORMS:
        t = t1 + t2
        assert close(t.inverse_transform(t.transform(new_points)), new_points), f"Transform {t} can not be inverted"

# Invert all compositions of transforms
for t1 in ALL_TRANSFORMS:
    for t2 in ALL_TRANSFORMS:
        t = t1 + t2
        assert close(t.transform(t.invert().transform(new_points)), new_points), f"Error with inverse of transform {t}"

# Test all compositions for compositionality
for t1 in ALL_TRANSFORMS:
    for t2 in ALL_TRANSFORMS:
        t = t1 + t2
        assert close(t2.transform(t1.transform(new_points)), t.transform(new_points)), f"Transform {t} did not compose"


# Test partial compositions for inversion identity
for t1 in ALL_TRANSFORMS:
    for t2 in _POINT_TRANSFORMS:
        _t = t1 + t2
        t = _t(points_pre, points_post)
        assert close(t.inverse_transform(t.transform(new_points)), new_points), f"Transform {t} was not close"
    for t2,t2p in zip(_FIXED_TRANSFORMS,_FIXED_TRANSFORMS_PARAMS):
        _t = t1 + t2
        t = _t(**t2p)
        assert close(t.inverse_transform(t.transform(new_points)), new_points), f"Transform {t} was not close"

# Test partial compositions for compositionality
for t1 in ALL_TRANSFORMS:
    for t2 in _POINT_TRANSFORMS:
        _t = t1 + t2
        t = _t(points_pre, points_post)
        assert close(t.transform(new_points), t2(points_pre, points_post).transform(t1.transform(new_points))), f"Transform {t} was not close"
    for t2,t2p in zip(_FIXED_TRANSFORMS,_FIXED_TRANSFORMS_PARAMS):
        _t = t1 + t2
        t = _t(**t2p)
        assert close(t.transform(new_points), t2(**t2p).transform(t1.transform(new_points))), f"Transform {t} was not close"


checkerboard = np.asarray([[[float(((i//10) + (j//10) + (k//10)) % 2) for i in range(0, 150)] for j in range(0, 300)] for k in range(0, 30)])

# Test compositions when transforming images
for t in ALL_TRANSFORMS:
    image_transformed_once = t.transform_image(checkerboard, relative=False)
    image_transformed_twice = t.transform_image(image_transformed_once, relative=False)
    image_transformed_sum = (t+t).transform_image(checkerboard, relative=False)
    corr = np.corrcoef(image_transformed_twice.flatten(), image_transformed_sum.flatten())[0,1]
    assert corr > .95, f"Correlation for composition of {t} was too low, it was {corr}"

# Test for exact answers for some transforms
assert close(TranslateFixed(z=5, y=4, x=7).transform(points_pre), points_pre+[5,4,7])
assert close(Identity().transform(points_pre), points_pre)
assert close(TranslateRotateFixed(z=3, y=8, x=-3, zrotate=8, yrotate=-9, xrotate=2).transform(points_pre), (points_pre+[3,8,-3])@rotation_matrix(8,-9,2))
assert close(Translate(points_pre, points_pre+[5,4,3]).transform(points_pre), points_pre+[5,4,3])
assert close(TranslateRotate(points_pre, (points_pre+[-4,2,1])@rotation_matrix(6,1,-3)).transform(points_pre), (points_pre+[-4,2,1])@rotation_matrix(6,1,-3))
assert close(TranslateRotate2D(points_pre, (points_pre+[0,2,1])@rotation_matrix(30,0,0)).transform(points_pre), (points_pre+[0,2,1])@rotation_matrix(30,0,0))
assert close(Rescale(z=3, y=1, x=.5).transform(points_pre), points_pre*[3,1,.5])

# We need to make some new transforms here to make sure the spot always stays
# within the image in both absolute and relative coordinates.
spot = np.zeros((80,90,100))
spotpos = (51,65,53)
spot[spotpos] = 1
_FIXED_TRANSFORMS_SPOT = [TranslateRotateFixed, TranslateFixed, Identity]
_FIXED_TRANSFORMS_PARAMS_SPOT = [dict(z=3.2, y=5, x=-24, zrotate=3.4, yrotate=5, xrotate=10), dict(z=-10, y=.3, x=4), dict()]
FIXED_TRANSFORMS_SPOT = [t(**tp) for t,tp in zip(_FIXED_TRANSFORMS_SPOT, _FIXED_TRANSFORMS_PARAMS_SPOT)]
_POINT_TRANSFORMS_SPOT = [TranslateRotate2D, Translate, TranslateRotate]
points_pre = np.random.randn(100,3)+50
points_post = (points_pre@rotation_matrix(1,2,3))-9
POINT_TRANSFORMS_SPOT = [t(points_pre, points_post) for t in _POINT_TRANSFORMS_SPOT]
ALL_TRANSFORMS_SPOT = FIXED_TRANSFORMS_SPOT + POINT_TRANSFORMS_SPOT

# Test image transformations in absolute and relative coordinates
for t in ALL_TRANSFORMS_SPOT:
    spot_rel = np.mean(np.where(t.transform_image(spot, relative=True)>.1), axis=1)
    assert np.max(spot_rel - (t.transform([spotpos]) - t.origin_and_maxpos(spot)[0])) < 1
    spot_abs = np.mean(np.where(t.transform_image(spot, relative=False)>.1), axis=1)
    assert np.max(spot_abs - t.transform([spotpos])) < 1

# Test compositions in absolute and relative coordinates
for t1 in ALL_TRANSFORMS_SPOT:
    for t2 in ALL_TRANSFORMS_SPOT:
        t = t1 + t2
        spot_rel = np.mean(np.where(t.transform_image(spot, relative=True)>.1), axis=1)
        assert np.max(spot_rel - (t.transform([spotpos]) - t.origin_and_maxpos(spot)[0])) < 1
        spot_abs = np.mean(np.where(t.transform_image(spot, relative=False)>.1), axis=1)
        assert np.max(spot_abs - t.transform([spotpos])) < 1



# Test nonrigid transformations
class _TranslateComplicated(Translate):
    """Should be identical to Translate, included for testing only."""
    def transform_image(self, *args, **kwargs):
        return Transform.transform_image(self, *args, **kwargs)

class _TranslateRotateComplicated(TranslateRotate):
    """Should be identical to TranslateRotate, included for testing only."""
    def transform_image(self, *args, **kwargs):
        return Transform.transform_image(self, *args, **kwargs)

for simple, complicated in [(Translate, _TranslateComplicated), (TranslateRotate, _TranslateRotateComplicated)]:
    im1 = complicated(points_pre, points_post).transform_image(checkerboard, relative=False)
    im2 = simple(points_pre, points_post).transform_image(checkerboard, relative=False)
    corr = np.corrcoef(im1.flatten(), im2.flatten())[0,1]
    assert corr > .95, f"Correlation for normal and complicated version of  {simple} was too low, it was {corr}"
    im1 = complicated(points_pre, points_post).transform_image(checkerboard, relative=True)
    im2 = simple(points_pre, points_post).transform_image(checkerboard, relative=True)
    corr = np.corrcoef(im1.flatten(), im2.flatten())[0,1]
    assert corr > .95, f"Correlation for normal and complicated version of  {simple} was too low with input bounds, it was {corr}"

# Test graphs
g = TransformGraph("mygraph")
g.add_node("a")
g.add_node("bx")
g.add_node("c", image=np.random.randn(2,3,4))
g.add_node("d")
g.add_edge("a", "bx", ALL_TRANSFORMS[0])
g.add_edge("a", "c", ALL_TRANSFORMS[1])
g.add_edge("c", "d", ALL_TRANSFORMS[2])
assert close(g.get_transform("bx", "d").transform(g.get_transform("d", "bx").transform(new_points)), new_points)
assert g.get_image("c").shape == (2,3,4)
assert g == g, "Self-equality failed"
with tempfile.TemporaryDirectory() as tmpdir:
    fn = Path(tmpdir).joinpath("file.npz")
    g.save(fn)
    g2 = TransformGraph.load(fn)
    assert g == g2
