import numpy as np

class ndarray_shifted(np.ndarray):
    def __new__(cls, a, origin=[0,0,0], only_if_necessary=False):
        if isinstance(a, cls):
            origin = a.origin
        if np.all(origin == np.asarray([0,0,0])) and only_if_necessary:
            return a
        arr = np.asarray(a).view(cls)
        if isinstance(origin, (float, int)):
            osigin = [origin]*a.ndim
        arr.origin = np.asarray(origin)
        # Finally, we must return the newly created object:
        return arr
    def __array_finalize__(self, obj):
        if obj is None: return
        self.origin = getattr(obj, 'origin', np.asarray([0,0,0]))
    def __repr__(self):
        s = super().__repr__()
        assert s[-1] == ")", "Cannot print"
        ret = s[:-1]
        if np.any(self.origin != [0,0,0]):
            ret += f", origin={list(self.origin)!r}"
        return ret+")"
