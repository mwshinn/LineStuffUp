import numpy as np

class ndarray_shifted(np.ndarray):
    def __new__(cls, a, origin=[0,0,0], downsample=[1,1,1]):
        arr = np.asarray(a).view(cls)
        arr.origin = np.asarray(origin)
        arr.downsample = np.asarray(downsample, dtype="int")
        # Finally, we must return the newly created object:
        return arr
    def __array_finalize__(self, obj):
        if obj is None: return
        self.origin = getattr(obj, 'origin', np.asarray([0,0,0]))
        self.downsample = getattr(obj, 'downsample', np.asarray([1,1,1], dtype="int"))
    def __repr__(self):
        s = super().__repr__()
        assert s[-1] == ")", "Cannot print"
        return s[:-1] + f", origin={self.origin}, downsample={self.downsample})"
