import numpy as np
import scipy
import SimpleITK as sitk

# TODO:
#
# - Allow composing transforms (for transforming)
# - Allow fitting one first and then the next
# - Unit tests


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

    Required methods to subclass: transform (map points from base space to the
    new space), inverse_transform (the opposite).  If parameters need to be
    calculated from the points, also define _fit (takes points_start and
    points_end and modifies the current object).

    If you want parameters to be accepted by the constructor, use the
    "DEFAULT_PARAMETERS" dict, where the key is the name and the value is the default.
    They will be saved in self.params.  This is NOT for parameters which need to
    be fit or can be reconstructed perfectly from points_start and points_end.

    """
    DEFAULT_PARAMETERS = {}
    def transform(self, points):
        raise NotImplementedError("Please subclass and replace")
    def inverse_transform(self, points):
        raise NotImplementedError("Please subclass and replace")
    def transform_relative(self, points):
        return self.transform(points) - self.origin
    def inverse_transform_relative(self, points):
        return self.inverse_transform(points + self.origin)
    def __init__(self, points_start=None, points_end=None, input_bounds=None, **kwargs):
        # Initialise parameters to either pass values or defaults
        self.params = {}
        for k in self.DEFAULT_PARAMETERS.keys():
            self.params[k] = kwargs[k] if k in kwargs else self.DEFAULT_PARAMETERS[k]
        # Check for invalid arguments
        for k in kwargs.keys():
            assert k in self.DEFAULT_PARAMETERS.keys(), f"Keyword argument {k} is not valid for the transform {type(self)}"
        # Save and process the points for the transform
        points_start = np.asarray(points_start)
        points_end = np.asarray(points_end)
        assert points_start.shape == points_end.shape, "Points start and end must be the same size"
        self.points_start = points_start
        self.points_end = points_end
        self.input_bounds = input_bounds
        # If this transform needs to be fit, then fit it.
        if hasattr(self, "_fit"):
            self._fit()
        if input_bounds is not None:
            corners_pretransform = [[a, b, c] for a in [0, self.input_bounds[0]] for b in [0, self.input_bounds[1]] for c in [0, self.input_bounds[2]]]
            self.origin = np.min(self.transform(corners_pretransform), axis=0)
            self.maxpos = np.max(self.transform(corners_pretransform), axis=0)
        else:
            self.origin = np.asarray([0, 0, 0])
            self.maxpos = None
    @classmethod
    def from_transform(cls, transform, *args, **kwargs):
        """Alternative constructor which steals the points from an existing Transform object"""
        return cls(points_start=transform.points_start, points_end=transform.points_end, input_bounds=transform.input_bounds, *args, **kwargs)
    def __repr__(self):
        ret = self.__class__.__name__
        ret += "("
        ret += f"points_start={self.points_start.tolist()}, points_end={self.points_end.tolist()}"
        if self.input_bounds is not None:
            ret += f", input_bounds={repr(self.input_bounds)}"
        for k,v in self.params.items():
            ret += ", {k}={v}"
        ret += ")"
        return ret
    def __add__(self, other):
        return _TransformComposed(self, other)
    def invert(self, input_bounds=None):
        """Invert the transform.

        Note: This won't actually work for all transforms.  Currently it just
        swaps the order of the points.  Need to be improved in the future.

        """
        return self.__class__(points_start=self.points_end, points_end=self.points_start, input_bounds=input_bounds, **self.params)
    def transform_image(self, img, relative=False):
        """Generic non-rigid transformation for images.

        Apply the transformation to image `img`.  `pad` is the number of pixels
        of zeros to pad on each side, it can be a scalar or a length-3 vector.
        (This way, transformations will not be clipped at the image boundaries.)

        This can be overridden by more efficient implementations in subclasses.

        """
        shape = np.round(np.ceil(self.maxpos - self.origin)).astype(int)
        # First, we construct a list of coordinates of all the pixels in the
        # image, and transform them to find out which point is mapped to which
        # other point.  Then, we inverse transform them to construct a matrix of
        # mappings.  We turn this matrix of mappings into a matrix of shifts
        # (displacements) by subtracting the initial coordinates.
        meshgrid = np.array(np.meshgrid(np.arange(0, img.shape[0]), np.arange(0,img.shape[1]), np.arange(0,img.shape[2]), indexing="ij"), dtype="float")
        grid = meshgrid.T.reshape(-1,3)
        mapped_grid = self.inverse_transform(grid+self.origin) - grid
        #recovered = grid.reshape(*img.shape[::-1],3).T
        #assert np.all(recovered == meshgrid)
        displacement = mapped_grid.reshape(*img.shape[::-1],3).T
        #displacement = mapped_img
        # Once we have the matrix of shifts/displacements, we convert it to a
        # "Vector image" (some data structure from the imagination of the ITK
        # developers) and convert the image we are transforming into ITK format
        # too.  We then use a bunch of magic I don't understand to apply the
        # transformation.  Really, I would have used a library other than ITK
        # but couldn't find any others which performed this transformation.
        sitk_displacement = sitk.GetImageFromArray(displacement[::-1].transpose(1,2,3,0)) # Haven't confirmed this transpose is right
        sitk_image = sitk.GetImageFromArray(img)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_image)
        dis_tx = sitk.DisplacementFieldTransform(sitk.Cast(sitk_displacement.ToVectorImage(),sitk.sitkVectorFloat64))
        #dis_tx = sitk.Euler3DTransform()
        #dis_tx.SetRotation(0,0,np.pi/4)
        resampler.SetTransform(dis_tx)
        res = resampler.Execute(sitk_image)
        res_img = sitk.GetArrayFromImage(res)
        return res_img

# class _TransformComposed(Transform):
#     def __init__(self, tf1, tf2):
#         self.tf1 = tf1
#         self.tf2 = tf2
#     def _fit(self):
#         pass
#     def __repr__(self):
#         return f"_TransformComposed(tf1={repr(tf1)}, tf2={repr(tf2)})"
#     def transform(self, points):
#         return self.tf2.transform(self.tf1.transform(points))
#     def inverse_transform(self, points):
#         return self.tf1.inverse_transform(self.tf2.inverse_transform(points))
#     def invert(self):
#         return self.__class__(tf2.invert(), tf1.invert())

class AffineBase(Transform):
    """A super-class for affine transforms.

    All subclasses implement a transformation which consists of a shift plus a
    matrix multiplication.

    To subclass: define a function "matrix(self, params)" which generates a
    transformation matrix.  Params should be a vector of parameters needed by
    the transformation matrix.  You don't need to add in the translation params
    because these are applied automatically.  Then, define DEFAULT_PARAMS to be
    the default arguments fed into `params` of the `matrix` method.  This also
    is how the class figures out how many parameters to pass to `matrix`.
    """ 
    def _fit(self):
        if self.points_start.shape[0] == 0:
            self.shift_params = np.zeros(3)
            self.matrix_params = np.zeros(len(self.DEFAULT_PARAMS))
            return
        # Do a special case for 2D transforms where we constrain z shift to be
        # 0.  We detect 2D from the input points, regardless of whether or not a
        # 2D transform is selected.  This allows 2D transforms to be performed
        # on 3D data but with a slightly improved optimization efficiency.
        if np.all(np.isclose(self.points_start[:,0], 0)) and np.all(np.isclose(self.points_end[:,0], 0)):
            # First three args are shift, rest of args are for transformation matrix
            def obj_func(params):
                return np.sum(np.square(self.points_end - (self.points_start - [0, params[0], params[1]]) @ self.matrix(params[2:])))
            self.optimize_result = scipy.optimize.minimize(obj_func, [0, 0] + self.DEFAULT_PARAMS)
            self.shift_params = np.concatenate([[0], self.optimize_result.x[0:2]])
            self.matrix_params = self.optimize_result.x[2:]
        else:
            # First three args are shift, rest of args are for transformation matrix
            def obj_func(params):
                return np.sum(np.square(self.points_end - (self.points_start - params[0:3]) @ self.matrix(params[3:])))
            self.optimize_result = scipy.optimize.minimize(obj_func, [0, 0, 0] + self.DEFAULT_PARAMS)
            self.shift_params = self.optimize_result.x[0:3]
            self.matrix_params = self.optimize_result.x[3:]
    def transform(self, points):
        points = np.asarray(points)
        if points.shape[0] == 0:
            return points
        return (points - self.shift_params) @ self.matrix(self.matrix_params)
    def inverse_transform(self, points):
        points = np.asarray(points)
        return points @ np.linalg.inv(self.matrix(self.matrix_params)) + self.shift_params
    def transform_image(self, image, relative=False):
        """A more efficient implementation for affine transforms"""
        if self.input_bounds is not None and relative is True:
            shape = np.round(np.ceil(self.maxpos - self.origin)).astype(int)
        else:
            shape = image.shape
        mat = np.linalg.inv(self.matrix(self.matrix_params)).T
        return scipy.ndimage.affine_transform(image, mat, self.shift_params + self.origin @ np.linalg.inv(mat) * np.linalg.det(mat), output_shape=shape)


def compose_affine_transforms(tf1, tf2):
    if tf1.__class__ != tf2.__class__:
        raise ValueError("Cannot compose, different types")
    return tf1.__class__(np.concatenate([tf2.points_start, tf2.inverse_transform(tf1.points_start)]),
                         np.concatenate([tf1.transform(tf2.points_end), tf1.points_end]))
    #return tf1.__class__(tf2.points_start,
    #                     tf1.transform(tf2.points_end))

class TranslateRotate(AffineBase):
    DEFAULT_PARAMS = [0, 0, 0]
    def matrix(self, params):
        return rotation_matrix(*params)

class TranslateRotate2D(AffineBase):
    DEFAULT_PARAMS = [0]
    def matrix(self, params):
        return rotation_matrix(params[0], 0, 0)

class Translate(AffineBase):
    DEFAULT_PARAMS = []
    def matrix(self, params):
        return np.eye(3)

class TranslateRotateRescaleUniform2D(AffineBase):
    DEFAULT_PARAMS = [0, 1]
    def _fit(self):
        """Too many parameters for the normal fitting routine"""
        if self.points_start.shape[0] == 0:
            self.shift_params = np.zeros(3)
            self.matrix_params = np.zeros(len(self.DEFAULT_PARAMS))
            return
        starting_transform = TranslateRotate2D(self.points_start, self.points_end)
        starting_params = list(starting_transform.shift_params) + list(starting_transform.matrix_params) + [1]
        # First three args are shift, rest of args are for transformation matrix
        def obj_func(params):
            return np.sum(np.square(self.points_end - (self.points_start - [0, params[0], params[1]]) @ self.matrix(params[2:])))
        self.optimize_result = scipy.optimize.minimize(obj_func, starting_params)
        self.shift_params = np.concatenate([[0], self.optimize_result.x[0:2]])
        self.matrix_params = self.optimize_result.x[2:]
    def matrix(self, params):
        return rotation_matrix(params[0], 0, 0) @ np.diag([1, params[1], params[1]])

class Identity(Transform):
    def transform(self, points):
        return points
    def inverse_transform(self, points):
        return points
    def transform_image(self, image, relative=False):
        """More efficient implementation of image transformation"""
        return image

class _TranslateComplicated(Translate):
    """Should be identical to Translate, included for testing only."""
    def transform_image(self, image):
        return Transform.transform_image(self, image)

class _TranslateRotateComplicated(TranslateRotate):
    """Should be identical to TranslateRotate, included for testing only."""
    def transform_image(self, image):
        return Transform.transform_image(self, image)

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


# TODO not so sure about this?
class Rescale(Transform):
    DEFAULT_PARAMETERS = {"scale": [1,1,1]}
    def transform(self, points):
        return points * np.asarray(self.params['scale'])
    def inverse_transform(self, points):
        return points / np.asarray(self.params['scale'])
    def invert(self):
        return self.__class__(self, scale=1/self.params['scale'])
    def transform_image(self, image):
        mat = np.eye(3)*scale
        zoomed = scipy.ndimage.zoom(image, self.params['scale'])
        return zoomed
        

from scipy.interpolate import griddata

# TODO finished but untested
class Triangulation(Transform):
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
