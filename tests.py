
from transform import *

_FIXED_TRANSFORMS = [TranslateRotateFixed, TranslateRotateFixed, TranslateFixed, Identity]
_FIXED_TRANSFORMS_PARAMS = [dict(z=3.2, y=5, x=-24, zrotate=3.4, yrotate=10, xrotate=20), dict(z=3.2, y=0, x=-24, zrotate=3.4, yrotate=10, xrotate=25), dict(z=-10, y=.3, x=4), dict()]
FIXED_TRANSFORMS = [t(**tp) for t,tp in zip(_FIXED_TRANSFORMS, _FIXED_TRANSFORMS_PARAMS)]
_POINT_TRANSFORMS = [TranslateRotate2D, Translate, TranslateRotate]
points_pre = np.random.randn(10,3)
points_post = points_pre@rotation_matrix(4,6,2)-9
POINT_TRANSFORMS = [t(points_pre, points_post) for t in _POINT_TRANSFORMS]

ALL_TRANSFORMS = FIXED_TRANSFORMS + POINT_TRANSFORMS

# Testing all transforms
new_points = np.random.randn(100,3)
for t in ALL_TRANSFORMS:
    assert np.allclose(t.inverse_transform(t.transform(new_points)), new_points), f"Transform {t} can not be inverted"

# Test all compositions for inversion identity
for t1 in ALL_TRANSFORMS:
    for t2 in ALL_TRANSFORMS:
        t = t1 + t2
        assert np.allclose(t.inverse_transform(t.transform(new_points)), new_points), f"Transform {t} can not be inverted"

# Test all compositions for compositionality
for t1 in ALL_TRANSFORMS:
    for t2 in ALL_TRANSFORMS:
        t = t1 + t2
        assert np.allclose(t2.transform(t1.transform(new_points)), t.transform(new_points)), f"Transform {t} did not compose"


# Test partial compositions for inversion identity
for t1 in ALL_TRANSFORMS:
    for t2 in _POINT_TRANSFORMS:
        _t = t1 + t2
        t = _t(points_pre, points_post)
        assert np.allclose(t.inverse_transform(t.transform(new_points)), new_points), f"Transform {t} was not close"
    for t2,t2p in zip(_FIXED_TRANSFORMS,_FIXED_TRANSFORMS_PARAMS):
        _t = t1 + t2
        t = _t(**t2p)
        assert np.allclose(t.inverse_transform(t.transform(new_points)), new_points), f"Transform {t} was not close"

# Test partial compositions for compositionality
for t1 in ALL_TRANSFORMS:
    for t2 in _POINT_TRANSFORMS:
        _t = t1 + t2
        t = _t(points_pre, points_post)
        assert np.allclose(t.transform(new_points), t2(points_pre, points_post).transform(t1.transform(new_points))), f"Transform {t} was not close"
    for t2,t2p in zip(_FIXED_TRANSFORMS,_FIXED_TRANSFORMS_PARAMS):
        _t = t1 + t2
        t = _t(**t2p)
        assert np.allclose(t.transform(new_points), t2(**t2p).transform(t1.transform(new_points))), f"Transform {t} was not close"


checkerboard = np.asarray([[[float(((i//10) + (j//10) + (k//10)) % 2) for i in range(0, 150)] for j in range(0, 300)] for k in range(0, 30)])

# Test compositions when transforming images
for t in ALL_TRANSFORMS:
    image_transformed_once = t.transform_image(checkerboard)
    image_transformed_twice = t.transform_image(image_transformed_once)
    image_transformed_sum = (t+t).transform_image(checkerboard)
    corr = np.corrcoef(image_transformed_twice.flatten(), image_transformed_sum.flatten())[0,1]
    assert corr > .95, f"Correlation for composition of {t} was too low, it was {corr}"


# TODO Test using the origin and relative coords
