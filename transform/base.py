import numpy as np
import scipy
from .ndarray_shifted import ndarray_shifted

# TODO:
# - implement posttransforms, allowing the unfitted transform to be on the left hand side

def rotation_matrix(z, y, x):
    """Perform *clockwise* rotation in degrees along the three axes"""
    sin = lambda x : np.sin(np.deg2rad(x))
    cos = lambda x : np.cos(np.deg2rad(x))
    zy_rotation = lambda theta : \
        np.asarray([[cos(theta), sin(theta), 0],
                    [-sin(theta), cos(theta), 0],
                    [0, 0, 1]])
    yx_rotation = lambda theta : np.roll(zy_rotation(theta), 1, axis=(0,1))
    xz_rotation = lambda theta : np.roll(zy_rotation(theta), -1, axis=(0,1))
    return yx_rotation(z) @ xz_rotation(y) @ zy_rotation(x)


class Transform:
    """Base class for all transforms.

    To use, instantiate and then call fit().

    Conceptually, there are two types of transforms: those that use points (see
    PointTransform) and those that don't.  Parameters can either be definitions
    of the transform (e.g., z-shift) or they can be hyperparameters (e.g., a
    smoothness regularizser).  They should be floats or booleans, and can be set
    through the GUI.

    Required methods to subclass: "transform" (map points from base space to the
    new space), "inverse_transform" (the opposite).  If parameters need to be
    calculated from the points, also define _fit (takes points_start and
    points_end and modifies the current object).  You should define the invert
    function if possible - if you do, inverse_transform will be automatically
    defined for you.

    If you want parameters to be accepted by the constructor, use the
    "DEFAULT_PARAMETERS" dict, where the key is the name and the value is the default.
    They will be saved in self.params.  This is NOT for parameters which need to
    be fit or can be reconstructed perfectly from points_start and points_end.

    There are a few final rules that must be followed when implementing new
    Transforms:

    1. If you don't define self.invert, it must return NotImplementedError (the default)
    2. The transform MUST be able to be reconstructed perfectly from the output of the __repr__ function.

    """
    DEFAULT_PARAMETERS = {}
    GUI_DRAG_PARAMETERS = [None, None, None]
    def __init__(self, **kwargs):
        # Initialise parameters to either pass values or defaults
        self.params = {}
        for k in self.DEFAULT_PARAMETERS.keys():
            self.params[k] = kwargs[k] if k in kwargs else self.DEFAULT_PARAMETERS[k]
        # Check for invalid arguments
        for k in kwargs.keys():
            assert k in self.DEFAULT_PARAMETERS.keys(), f"Keyword argument {k} is not valid for the transform {type(self)}"
        # If this transform needs to be fit, then fit it.
        if hasattr(self, "_fit"):
            self._fit()
    def __repr__(self):
        ret = self.__class__.__name__
        ret += "("
        parts = []
        for k,v in self.params.items():
            parts.append(f"{k}={v}")
        ret += ", ".join(parts)
        ret += ")"
        return ret
    def __eq__(self, other):
        return repr(self) == repr(other)
    def __add__(self, other):
        return compose_transforms(self, other)
    def transform(self, points):
        """Forward mapping function for the transformation"""
        raise NotImplementedError("Please subclass and replace")
    def inverse_transform(self, points):
        """Inverse mapping function for the transformation.

        Override this function to provide a more efficient implementation.
        """
        return self.invert().transform(points)
    def invert(self):
        raise NotImplementedError("Please subclass and replace")
    def origin_and_maxpos(self, img):
        """When using relative mode for image transformation, find the corners of the bounds based on the input image size"""
        input_bounds = img.shape
        origin_offset = img.origin if isinstance(img, ndarray_shifted) else [0,0,0]
        print(origin_offset)
        if input_bounds is not None:
            corners_pretransform = [[a, b, c] for a in [0, input_bounds[0]] for b in [0, input_bounds[1]] for c in [0, input_bounds[2]]]
            corners_pretransform = corners_pretransform + np.asarray(origin_offset)
            origin = np.min(self.transform(corners_pretransform), axis=0)
            maxpos = np.max(self.transform(corners_pretransform), axis=0)
        else:
            origin = np.asarray([0, 0, 0])
            maxpos = None
        return origin,maxpos
    def transform_image(self, img, relative=True, labels=False):
        """Generic non-rigid transformation for images.

        Apply the transformation to image `img`.  `pad` is the number of pixels
        of zeros to pad on each side, it can be a scalar or a length-3 vector.
        (This way, transformations will not be clipped at the image boundaries.)

        If `labels` is True, no interpolation is performed.

        This can be overridden by more efficient implementations in subclasses.

        """
        origin, maxpos = self.origin_and_maxpos(img)
        if relative:
            shape = np.round(np.ceil(maxpos - origin)).astype(int)
        else:
            shape = np.asarray(img.shape).astype(int)
            origin = np.zeros(3)
        origin_adjust = img.origin if isinstance(img, ndarray_shifted) else np.asarray([0,0,0])
        # First, we construct a list of coordinates of all the pixels in the
        # image, and transform them to find out which point is mapped to which
        # other point.  Then, we inverse transform them to construct a matrix of
        # mappings.  We turn this matrix of mappings into a matrix of pointers
        # from the destination image to the source image, and then use the
        # map_coordinates function to perform this mapping.
        meshgrid = np.array(np.meshgrid(np.arange(0, shape[0]), np.arange(0,shape[1]), np.arange(0,shape[2]), indexing="ij"), dtype="float")
        grid = meshgrid.T.reshape(-1,3)
        mapped_grid = self.inverse_transform(grid+origin)-origin_adjust
        displacement = mapped_grid.reshape(*shape[::-1],3).T
        # Prefilter == False speeds it up by about 20%.  Supposedly it makes the
        # output images blurrier though, having't done a comparison yet.
        order = 0 if labels else 3
        return ndarray_shifted(scipy.ndimage.map_coordinates(img, displacement, prefilter=(not labels), order=order), origin=origin)
    @staticmethod
    def pretransform(*args, **kwargs):
        """Default fixed transform, applied before this transform is applied.

        This can usually be set to the Identity transformation, except when only
        part of the transform should be fit with data, for instance, composed
        transforms where only the last element is set to be fit.  This is also
        the default when the parameters for the transform have not yet been set.

        """
        return Identity()

class PointTransform(Transform):
    """Transformation based on starting and ending points

    Please define:
    - self.transform
    - self.inverse_transform
    - (Optionally) self.invert

    Guarantees access to:
    - self.points_start
    - self.points_end
    """
    def __init__(self, points_start=None, points_end=None, **kwargs):
        # Save and process the points for the transform
        points_start = np.asarray(points_start)
        points_end = np.asarray(points_end)
        assert points_start.shape == points_end.shape, "Points start and end must be the same size"
        self.points_start = points_start
        self.points_end = points_end
        super().__init__(**kwargs)
    @classmethod
    def from_transform(cls, transform, *args, **kwargs):
        """Alternative constructor which steals the points from an existing Transform object"""
        return cls(points_start=transform.points_start, points_end=transform.points_end, *args, **kwargs)
    def __repr__(self):
        ret = self.__class__.__name__
        ret += "("
        ret += f"points_start={self.points_start.tolist()}, points_end={self.points_end.tolist()}"
        for k,v in self.params.items():
            ret += f", {k}={v}"
        ret += ")"
        return ret
    def invert(self):
        """Invert the transform.

        Note: This will return incorrect results for some non-affine transforms.
        Currently it just swaps the order of the points.

        """
        return self.__class__(points_start=self.points_end, points_end=self.points_start, **self.params)

class AffineTransform:
    """AffineTransform should always be inherited with PointTransform

    To subclass, use the _fit function to define the parameters self.shift and
    self.matrix for the two components of the affine transform.  This defines
    all other necessary functions for a transform.  If you have multiple
    inheritance from PointTransform, you can use self.points_start and
    self.points_end.

    """
    def transform(self, points):
        points = np.asarray(points)
        if points.shape[0] == 0:
            return points
        return (points - self.shift) @ self.matrix
    def inverse_transform(self, points):
        points = np.asarray(points)
        if points.shape[0] == 0:
            return points
        return points @ np.linalg.inv(self.matrix) + self.shift


class TranslateRotate(AffineTransform,PointTransform):
    def _fit(self):
        demeaned_start = self.points_start - np.mean(self.points_start, axis=0)
        demeaned_end = self.points_end - np.mean(self.points_end, axis=0)
        U,S,V = np.linalg.svd(demeaned_start.T @ demeaned_end)
        self.matrix = U@V
        self.shift = np.mean(self.points_start - self.points_end@np.linalg.inv(self.matrix), axis=0)

class TranslateRotate2D(AffineTransform,PointTransform):
    def _fit(self):
        demeaned_start = self.points_start - np.mean(self.points_start, axis=0)
        demeaned_end = self.points_end - np.mean(self.points_end, axis=0)
        U,S,V = np.linalg.svd(demeaned_start[:,1:3].T @ demeaned_end[:,1:3])
        corner_matrix = U@V
        self.matrix = np.vstack([[[1, 0, 0]], np.hstack([[[0],[0]], corner_matrix])])
        self.shift = np.mean(self.points_start - self.points_end@np.linalg.inv(self.matrix), axis=0)

class Translate(AffineTransform,PointTransform):
    def _fit(self):
        self.matrix = np.eye(3)
        self.shift = np.mean(self.points_start - self.points_end, axis=0)

class Flip(AffineTransform,Transform):
    DEFAULT_PARAMETERS = {"z": False, "y": False, "x": False, "zthickness": 0, "ythickness": 0, "xthickness": 0}
    def _fit(self):
        sign = lambda x : -1 if self.params[x] else 1
        self.matrix = np.asarray([[sign("z"), 0, 0], [0, sign("y"), 0], [0, 0, sign("x")]])
        self.shift = np.asarray([max(0, self.params[c+"thickness"]-1)*int(self.params[c]) for c in ["z", "y", "x"]])
    def invert(self):
        return self

class TranslateFixed(AffineTransform,Transform):
    DEFAULT_PARAMETERS = {"z": 0.0, "y": 0.0, "x": 0.0}
    GUI_DRAG_PARAMETERS = ["z", "y", "x"]
    def _fit(self):
        self.matrix = np.eye(3)
        self.shift = np.asarray([-self.params["z"], -self.params["y"], -self.params["x"]])
    def invert(self):
        return self.__class__(x=-self.params["x"], y=-self.params["y"], z=-self.params["z"])

class TranslateRotateFixed(AffineTransform,Transform):
    DEFAULT_PARAMETERS = {"z": 0.0, "y": 0.0, "x": 0.0, "zrotate": 0.0, "yrotate": 0.0, "xrotate": 0.0, "invert": False}
    GUI_DRAG_PARAMETERS = ["z", "y", "x"]
    def _fit(self):
        self.matrix = rotation_matrix(self.params["zrotate"], self.params["yrotate"], self.params["xrotate"])
        if self.params['invert']:
            self.matrix = self.matrix.T
        self.shift = np.asarray([-self.params["z"], -self.params["y"], -self.params["x"]])
    def invert(self):
        newzyx = self.matrix.T @ [self.params["z"], self.params["y"], self.params["x"]]
        return self.__class__(zrotate=self.params["zrotate"], yrotate=self.params["yrotate"], xrotate=self.params["xrotate"], z=-newzyx[0], y=-newzyx[1], x=-newzyx[2], invert=True)
        

# class ZSlice(AffineTransform,Transform):
#     DEFAULT_PARAMETERS = {"zshift": 0, "yshift": 0, "xshift": 0, "yslope": 0, "xslope": 0}
#     def _fit(self):
#         self.shift = [self.params["zshift"], self.params["yshift"], self.params["xshift"]]
#         self.matrix = np.asarray([[1, -self.params["yslope"], -self.params["xslope"]], [0, 1, 0], [0, 0, 1]])

# class TranslateRotateRescaleUniform2D(AffineTransform,PointTransform):
#     def _fit(self):
#         """Too many parameters for the normal fitting routine"""
#         if self.points_start.shape[0] == 0:
#             self.shift = np.zeros(3)
#             self.matrix = np.zeros(len(self.DEFAULT_PARAMS))
#             return
#         starting_transform = TranslateRotate2D(self.points_start, self.points_end)
#         starting_params = list(starting_transform.shift) + list(starting_transform.matrix) + [1]
#         print(starting_transform, starting_params)
#         # First three args are shift, rest of args are for transformation matrix
#         def obj_func(params):
#             return np.sum(np.square(self.points_end - (self.points_start - [0, params[0], params[1]]) @ self._matrix(params[2:])))
#         self.optimize_result = scipy.optimize.minimize(obj_func, starting_params)
#         self.shift = np.concatenate([[0], self.optimize_result.x[0:2]])
#         self.matrix = self._matrix(self.optimize_result.x[2:])
#     def _matrix(self, params):
#         return rotation_matrix(params[0], 0, 0) @ np.diag([1, params[1], params[1]])

class Identity(AffineTransform,Transform):
    def _fit(self):
        self.matrix = np.eye(3)
        self.shift = np.zeros(3)
    def transform(self, points):
        return points
    def inverse_transform(self, points):
        return points
    def invert(self):
        return self.__class__()
    def transform_image(self, image, relative=False, labels=False):
        """More efficient implementation of image transformation"""
        return image

# class TransformSquareWeightedInterpolation(Transform):
#     def _fit(before, after):
#         from sklearn.linear_model import LinearRegression
#         self.reg = LinearRegression().fit(before, after)
#         self.vectors = after - reg.predict(before)
#         self.before = before
#     def transform(points):
#         dists = scipy.spatial.distance_matrix(points, self.before)
#         pull = 1/dists**2
#         weights = pull/np.sum(pull, axis=1, keepdims=True)
#         weights[np.isnan(weights)] = 1
#         transform = weights @ self.vectors
#         return self.reg.predict(points)+transform
#     def inverse_transform(self, points):


# class TransformTriangulation2D(Transform):
#     DEFAULT_PARAMETERS = {"fixed_axis": 0}
#     def _fit(self, before, after):
#         self.fixed_axis = self.parameters["fixed_axis"]
#         self.fit_axes = list(sorted(set([0, 1, 2])-set([self.fixed_axis])))
#         # Assume axes is (1,2)
#         self.triangulation = scipy.spatial.Delaunay(before[1:3])
#         self.triangulation_points_after = after[1:3]
#         self.affine_transforms = []
#         for s in self.triangulation.simplices:
#             x = self.triangulation.points[s]
#             y = self.triangulation_points_after[s]


class Rescale(AffineTransform,Transform):
    DEFAULT_PARAMETERS = {"z": 1.0, "y": 1.0, "x": 1.0}
    def _fit(self):
        self.matrix = np.diag([self.params["z"], self.params["y"], self.params["x"]])
        self.shift = np.asarray([0, 0, 0])
    def invert(self):
        return self.__class__(z=1/self.params["z"], y=1/self.params["y"], x=1/self.params["x"])

from scipy.interpolate import griddata

class Triangulation(PointTransform):
    def _fit(self):
        # To avoid out of bounds, we add a few pseudo points.  We do this by
        # finding the convex hull, centering it, scaling it, and then shifting
        # the scaled points back from the centering.  We assign these points a
        # simple linear transformed version of the points they are derived from.
        SCALE_FACTOR = 1000
        before = self.points_start
        after = self.points_end
        t = scipy.spatial.Delaunay(before) # Triangulation
        assert np.all(t.points == before), "Coplannar points"
        hull_points_inds = np.unique(t.convex_hull.flatten())
        hull_points_vecs = after[hull_points_inds] - before[hull_points_inds]
        hull_mean_shift = np.mean(before[hull_points_inds], axis=0)
        self.pseudopoints_start = SCALE_FACTOR*(before[hull_points_inds] - hull_mean_shift) + hull_mean_shift
        self.pseudopoints_end = self.pseudopoints_start + hull_points_vecs
        self.all_points_start = np.concatenate([self.points_start, self.pseudopoints_start])
        self.all_points_end = np.concatenate([self.points_end, self.pseudopoints_end])
    def transform(self, points):
        zcoords = griddata(self.all_points_start, self.all_points_end[:,0], points) 
        ycoords = griddata(self.all_points_start, self.all_points_end[:,1], points) 
        xcoords = griddata(self.all_points_start, self.all_points_end[:,2], points) 
        return np.concatenate([zcoords[:,None], ycoords[:,None], xcoords[:,None]], axis=1)
    def inverse_transform(self, points):
        zcoords = griddata(self.all_points_end, self.all_points_start[:,0], points) 
        ycoords = griddata(self.all_points_end, self.all_points_start[:,1], points) 
        xcoords = griddata(self.all_points_end, self.all_points_start[:,2], points) 
        return np.concatenate([zcoords[:,None], ycoords[:,None], xcoords[:,None]], axis=1)

def compose_transforms(a, b):
    # Special cases for linear and for adding to a class (not yet fitted)
    if isinstance(a, AffineTransform) and isinstance(b, AffineTransform):
        if isinstance(b, PointTransform):
            return compose_transforms(a, b.__class__)(points_start=b.points_start, points_end=b.points_end, **b.params)
        else:
            return compose_transforms(a, b.__class__)(**b.params)
    if isinstance(a, Transform) and isinstance(b, Transform):
        return Composed(a, b)
    if isinstance(a, Transform) and not isinstance(b, Transform):
        inherit = PointTransform if issubclass(b, PointTransform) else Transform
        if isinstance(a, AffineTransform) and issubclass(b, AffineTransform):
            class ComposedPartialAffine(AffineTransform,inherit):
                DEFAULT_PARAMETERS = b.DEFAULT_PARAMETERS # b.params # Changed from b.DEFAULT_PARAMETERS
                GUI_DRAG_PARAMETERS = b.GUI_DRAG_PARAMETERS
                def __init__(self, points_start=None, points_end=None, *args, **kwargs):
                    extra_args = {}
                    if points_start is not None and points_end is not None:
                        extra_args['points_start'] = points_start
                        extra_args['points_end'] = points_end
                    self.b_type = b
                    self.b = b(*args, **kwargs, **extra_args)
                    super().__init__(*args, **kwargs, **extra_args)
                def _fit(self):
                    self.matrix = a.matrix @ self.b.matrix
                    self.shift = a.shift + self.b.shift @ np.linalg.inv(a.matrix)
                def __repr__(self):
                    return repr(a) + " + " + repr(self.b)
                @staticmethod
                def pretransform(*args, **kwargs):
                    return a
                def invert(self):
                    return self.b.invert() + a.invert()
            return ComposedPartialAffine
        else:
            class ComposedPartial(inherit):
                DEFAULT_PARAMETERS = b.params if hasattr(b, "params") else b.DEFAULT_PARAMETERS #  b.params # Changed from b.DEFAULT_PARAMETERS
                GUI_DRAG_PARAMETERS = b.GUI_DRAG_PARAMETERS
                def __init__(self, points_start=None, points_end=None, *args, **kwargs):
                    extra_args = {}
                    if points_start is not None and points_end is not None:
                        extra_args['points_start'] = points_start
                        extra_args['points_end'] = points_end
                    self.b_type = b
                    self.b = b(*args, **kwargs, **extra_args)
                    super().__init__(**extra_args)
                def transform(self, points):
                    return self.b.transform(a.transform(points))
                def inverse_transform(self, points):
                    return a.inverse_transform(self.b.inverse_transform(points))
                def invert(self):
                    raise NotImplementedError
                def __repr__(self):
                    return repr(a) + " + " + repr(self.b)
                def invert(self):
                    return self.b.invert() + a.invert()
                @staticmethod
                def pretransform(*args, **kwargs):
                    return a
            return ComposedPartial
    raise NotImplementedError("Invalid composition")


