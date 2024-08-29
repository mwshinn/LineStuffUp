import numpy as np
import io
import imageio
import imageio.plugins.ffmpeg # If this fails, install the imageio-ffmpeg package with pip
import numcodecs

class WebM(numcodecs.abc.Codec):
    """Codec providing WebM lossy compression for zarr"""
    codec_id = "WebM"
    def __init__(self, bitrate=20000000, minval=0, maxval=255, transform="none"):
        self.transform = transform
        self.bitrate = bitrate
        self.minval = self._image_compression_transform(minval)
        self.maxval = self._image_compression_transform(maxval)
        super().__init__()
    def get_config(self):
        return {"id": self.codec_id, "bitrate": self.bitrate, "minval": self.minval, "maxval": self.maxval, "transform": self.transform}
    def encode(self, buf):
        buf = numcodecs.compat.ensure_ndarray(buf)
        assert len(buf.shape) == 3, "Only currently supported for 3D volumes"
        assert buf.shape[1] % 16 == 0, f"Last dim must be divisible by 16, invalid shape {buf.shape}"
        assert buf.shape[2] % 16 == 0, f"Last dim must be divisible by 16, invalid shape {buf.shape}"
        pseudofile = io.BytesIO()
        writer = imageio.get_writer(pseudofile, format="webm", fps=30, bitrate=self.bitrate, codec="vp9", macro_block_size=16)
        for p in buf:
            writer.append_data((255*(self._image_compression_transform(p)-self.minval)/(self.maxval-self.minval)).astype("uint8"))
        writer.close()
        return np.frombuffer(pseudofile.getvalue(), dtype=np.uint8)
    def decode(self, buf):
        buf = numcodecs.compat.ensure_ndarray(buf)
        pseudofile = io.BytesIO(buf.tobytes())
        r = imageio.get_reader(pseudofile, format="webm")
        d = np.asarray([it[:,:,0] for it in r.iter_data()], dtype="uint8")
        r.close()
        return self._image_decompression_transform(d.astype("float32")/255*(self.maxval-self.minval)+self.minval)
    def _image_compression_transform(self, img):
        if self.transform == "none": # None
            return img
        if self.transform == "log10": # Truncated log + 10
            return np.log(10+np.maximum(0, img))
        raise ValueError(f"Invalid transform {self.transform}")
    def _image_decompression_transform(self, img):
        if self.transform == "none": # None
            return img
        if self.transform == "log10": # Truncated log + 10
            return np.exp(img)-10
        raise ValueError(f"Invalid transform {self.transform}")

numcodecs.registry.register_codec(WebM)


# import zarr
# arr = zarr.creation.create(compressor=WebM(), shape=(100,1600,1600), chunks=(100,128,128), dtype='uint8')
# arr[:,40:80,40:80] = np.random.rand(100,40,40)*255

# import zarr
# arr = zarr.creation.create(compressor=WebM(maxval=1), shape=(100,1600,1600), chunks=(100,256,256), dtype='float32')
# dat = np.random.rand(100,40,40)
# arr[:,40:80,40:80] = dat
# arr[:,40:80,40:80]
