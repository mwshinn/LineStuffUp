import imageio
import numpy as np
import io
import scipy.stats
import imageio.plugins.ffmpeg # If this fails, install the imageio-ffmpeg package with pip
from .ndarray_shifted import ndarray_shifted

def apply_transform_to_2D_colour_image(image_filename, transform, flip=False):
    im = imageio.imread(image_filename).transpose(2,0,1)
    newim = np.zeros(im.shape)
    if flip:
        im = im[:,::-1]
    for i in range(0, im.shape[0]):
        newim[i] = transform.transform_image(im[i][None])
    filename_parts = image_filename.split(".")
    filename_parts.insert(-1, "transform")
    try:
        imageio.imsave(".".join(filename_parts), newim.transpose(1,2,0))
    except TypeError:
        newim = newim.transpose(1,2,0)
        newim -= np.min(newim)
        newim /= np.max(newim)
        newim = (255*newim).astype("uint8")
        imageio.imsave(".".join(filename_parts), newim)
        

def blit(source, target, loc):
    source_size = np.asarray(source.shape)
    target_size = np.asarray(target.shape)
    # If we had infinite boundaries, where would we put it?
    target_loc_tl = loc
    target_loc_br = target_loc_tl + source_size
    # Compute the index for the source
    source_loc_tl = -np.minimum(0, target_loc_tl)
    source_loc_br = source_size - np.maximum(0, target_loc_br - target_size)
    # Recompute the index for the target
    target_loc_br = np.minimum(target_size, target_loc_tl+source_size)
    target_loc_tl = np.maximum(0, target_loc_tl)
    # Compute slices from positions
    target_slices = [slice(s1, s2) for s1,s2 in zip(target_loc_tl,target_loc_br)]
    source_slices = [slice(s1, s2) for s1,s2 in zip(source_loc_tl,source_loc_br)]
    # Perform the blit
    target[tuple(target_slices)] = source[tuple(source_slices)]

def bake_images(im_fixed, im_movable, transform):
    origin = transform.origin_and_maxpos(im_movable)[0]
    ti = transform.transform_image(im_movable, relative=True)
    new_dims_max = np.ceil(np.max([ti.shape + origin, im_fixed.shape], axis=0)).astype(int)
    new_dims_min = np.floor(np.min([origin, [0,0,0]], axis=0)).astype(int)
    im = np.zeros(new_dims_max-new_dims_min, dtype=float)
    print(new_dims_max, new_dims_min, im.shape)
    blit(im_fixed, im, tuple([0,0,0]-new_dims_min))
    blit(im_movable, im, tuple(origin.astype(int)-new_dims_min))
    return ndarray_shifted(im, origin=new_dims_min)

def load_image(fn, channel=None):
    img = imageio.imread(fn)
    if channel is None:
        axes = list(np.any((img!=0) & (img!=255), axis=(0,1)))
        return np.mean(img[:,:,axes], axis=2)[None]
    else:
        return img[:,:,channel][None]

def image_is_label(img):
    """Quick, non-robust way to guess whether an image is a label image"""
    plane = img[img.shape[0]//2] # Pick a plane in the middle
    # First a quick test to eliminate most cases quickly
    pmini = plane[0:100,0:100]
    if len(np.unique(pmini)) > len(pmini.flat)/2:
        return False
    # Now a more complete test
    vals,counts = np.unique(plane, return_counts=True)
    if len(vals) > plane.shape[0]*plane.shape[1] / 25: # There are too many "labels"
        return False
    if not np.all(np.isclose(vals, vals.astype(int))): # Don't fall on integer values
        return False
    if len(vals) == 1: # All black
        return False
    if 0 not in vals or np.max(counts) != counts[vals==0][0]: # Zero isn't the most common
        return False
    return True

def _image_compression_transform(img, transform_id):
    if transform_id == 0: # None
        return img
    if transform_id == 1: # Truncated log + 10
        return np.log(10+np.maximum(0, img))

def _image_decompression_transform(img, transform_id):
    if transform_id == 0: # None
        return img
    if transform_id == 1: # Truncated log + 10
        return np.exp(img)-10

def _image_detect_transform(img):
    if scipy.stats.skew(img.flatten()) > 25: # Disable this for now
        return 1 # Truncated log + 10
    return 0 # None

def compress_image(img, level="normal"):
    assert level in ["low", "normal", "high"], "Invalid level"
    # Format code 0 == no compression
    # Format code 1 == vp9 video stack
    # Format code 2 == jpegs
    if img.ndim == 2:
        img = np.asarray([img])
    transform_id = _image_detect_transform(img)
    img = _image_compression_transform(img, transform_id)
    if image_is_label(img):
        return img, [0]
    if min(img.shape) > 10: # Compress volumes as a video in vp9 format (format code 1)
        bitrate = 20000000 if level == "normal" else 40000000 if level == "high" else 10000000
        maxplanes = np.quantile(img, .999)
        minplanes = np.min(img)
        img = np.minimum(img, maxplanes)
        print(maxplanes, minplanes)
        imgnorm = ((img-minplanes)/(maxplanes-minplanes)*255).astype("uint8")
        zdim = np.argmin(imgnorm.shape) # Thin dimension may not be z
        imgnorm = imgnorm.swapaxes(zdim, 0)
        # We need to make the image a size multiple of 16
        pady = 16 - (imgnorm.shape[1] % 16) % 16
        padx = 16 - (imgnorm.shape[2] % 16) % 16
        imgnorm = np.pad(imgnorm, ((0,0), (0,pady), (0,padx)))
        print("Newscale", imgnorm.shape, pady, padx, img.shape, zdim)
        kind = [1, transform_id, bitrate, maxplanes, minplanes, pady, padx, zdim]
        pseudofile = io.BytesIO()
        writer = imageio.get_writer(pseudofile, format="webm", fps=30, bitrate=bitrate, codec="vp9", macro_block_size=16)
        for p in imgnorm:
            writer.append_data(p)
        writer.close()
        return np.frombuffer(pseudofile.getvalue(), dtype=np.uint8), kind
    else: # Compress as jpegs (format code 2)
        quality = 90 if level == "normal" else 95 if level == "high" else 80
        files = []
        maxes = []
        mins = []
        for i in range(0, img.shape[0]):
            pseudofile = io.BytesIO()
            maxval = np.quantile(img[i], .999)
            minval = np.min(img[i])
            maxes.append(maxval)
            mins.append(minval)
            im = ((np.minimum(maxval, img[i])-minval)/(maxval-minval)*255).astype("uint8")
            imageio.v3.imwrite(pseudofile, im, format_hint=".jpeg", quality=quality)
            files.append(np.frombuffer(pseudofile.getvalue(), dtype=np.uint8))
        lens = list(map(len, files))
        print(lens, maxes, mins)
        info = np.concatenate(list(zip(lens, maxes, mins)))
        kind = [2, transform_id, quality]+info.tolist()
        return np.concatenate(files), kind

def decompress_image(data, kind):
    if kind[0] == 0:
        return data
    if kind[0] == 1:
        _,transform_id,bitrate,maxval,minval,pady,padx,zdim = kind
        print(maxval, minval)
        padx = int(padx)
        pady = int(pady)
        pseudofile = io.BytesIO(data.tobytes())
        r = imageio.get_reader(pseudofile, format="webm")
        d = np.asarray([it[:(it.shape[0]-pady),:(it.shape[1]-padx),0] for it in r.iter_data()], dtype="float32")
        d = d.swapaxes(int(zdim), 0)
        r.close()
        return _image_decompression_transform(d/255.0*(maxval-minval)+minval, int(transform_id))
    if kind[0] == 2:
        transform_id,quality = kind[1:3]
        lens = np.asarray(kind[3::3]).astype(int)
        maxes = kind[4::3]
        mins = kind[5::3]
        print(maxes, mins)
        ibase = 0
        ims = []
        for i,l in enumerate(lens):
            pseudofile = io.BytesIO(data[ibase:(ibase+l)].tobytes())
            im = np.asarray(imageio.v3.imread(pseudofile, format_hint=".jpeg"))
            im = _image_decompression_transform(im/255.0*(maxes[i]-mins[i])+mins[i], int(transform_id))
            ims.append(im)
            ibase += l
        return np.asarray(ims, dtype="float32")

