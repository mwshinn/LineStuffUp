import imageio
import numpy as np
import io
import scipy.stats
import zlib
import imageio.plugins.ffmpeg # If this fails, install the imageio-ffmpeg package with pip
import skimage
import skimage.registration
from .ndarray_shifted import ndarray_shifted

try: # Work around skimage bug in some versions
    phase_correlation = lambda x,y : skimage.registration.phase_cross_correlation(x, y, normalization=None)
    phase_correlation(np.asarray([1]), np.asarray([1]))
except TypeError:
    phase_correlation = lambda x,y : skimage.registration.phase_cross_correlation(x, y)

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
    blit(im_fixed, im, tuple([0,0,0]-new_dims_min))
    blit(im_movable, im, tuple(origin.astype(int)-new_dims_min))
    return ndarray_shifted(im, origin=new_dims_min)

def absolute_coords_to_voxel_coords(img, coords):
    if not isinstance(img, ndarray_shifted):
        img = ndarray_shifted(img)
    return np.round((coords - img.origin)/img.scale).astype(int)

def voxel_coords_to_absolute_coords(img, coords):
    if not isinstance(img, ndarray_shifted):
        img = ndarray_shifted(img)
    return coords * img.scale + img.origin

def crop_to_intersection(img1, img2):
    # TODO DOes not yet work with downsampling
    if not isinstance(img1, ndarray_shifted):
        img1 = ndarray_shifted(img1)
    if not isinstance(img2, ndarray_shifted):
        img2 = ndarray_shifted(img2)
    absolute_coords_to_voxel_coords = lambda img,coords: np.round((coords - img.origin)/img.scale).astype(int)
    voxel_coords_to_absolute_coords = lambda img,coords: coords * img.downsample + img.origin
    origin = np.max([img1.origin, img2.origin], axis=0)
    maxpos = np.min([voxel_coords_to_absolute_coords(img1, img1.shape), voxel_coords_to_absolute_coords(img2, img2.shape)], axis=0)
    output_img = np.zeros(img1.shape)
    i1 = absolute_coords_to_voxel_coords(img1, origin)
    i2 = absolute_coords_to_voxel_coords(img1, maxpos)
    img1_crop = np.array(img1)[i1[0]:i2[0],i1[1]:i2[1],i1[2]:i2[2]]
    i1 = absolute_coords_to_voxel_coords(img2, origin)
    i2 = absolute_coords_to_voxel_coords(img2, maxpos)
    img2_crop = np.array(img2)[i1[0]:i2[0],i1[1]:i2[1],i1[2]:i2[2]]
    return ndarray_shifted(img1_crop, origin), ndarray_shifted(img2_crop, origin)

    

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
    if np.median(np.abs(np.diff(plane, axis=0))) >= 1: # Should be mostly flat
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
    _img = img if np.prod(img.shape) < 10000000 else img[::4,::4,::4] # Hack for big images
    if scipy.stats.skew(_img.reshape(-1)) > 25:
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
    if False: # Image code 0 is uncompressed, which we don't use anymore.
        return img, [0]
    if image_is_label(img): # Lossless compression with gzip (format code 3)
        if np.max(img) < 2**8 and np.min(img) >= 0:
            img = img.astype("uint8")
        elif np.max(img) < 2**16 and np.min(img) >= 0:
            img = img.astype("uint16")
        # Gzip is fast but not great, so we compress twice and this works well (but why?)
        comp = zlib.compress(zlib.compress(img, 9), 9)
        return comp, [3, str(img.dtype), *img.shape]
    if min(img.shape) > 10: # Compress volumes as a video in vp9 format (format code 1)
        bitrate = 20000000 if level == "normal" else 40000000 if level == "high" else 10000000
        # We normalise in a complicated way to reduce memory usage for large images
        maxplanes = np.quantile(img, .999)
        minplanes = np.min(img)
        imgnorm = img.copy()
        imgnorm[imgnorm>maxplanes] = maxplanes
        imgnorm -= minplanes
        for i in range(0, imgnorm.shape[0]):
            imgnorm[i] = imgnorm[i]/(maxplanes-minplanes)*255
        imgnorm = imgnorm.astype("uint8")
        zdim = np.argmin(imgnorm.shape) # Thin dimension may not be z
        imgnorm = imgnorm.swapaxes(zdim, 0)
        # We need to make the image a size multiple of 16
        pady = 16 - (imgnorm.shape[1] % 16) % 16
        padx = 16 - (imgnorm.shape[2] % 16) % 16
        imgnorm = np.pad(imgnorm, ((0,0), (0,pady), (0,padx)))
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
        info = np.concatenate(list(zip(lens, maxes, mins)))
        kind = [2, transform_id, quality]+info.tolist()
        return np.concatenate(files), kind

def decompress_image(data, kind):
    if int(kind[0]) == 0:
        return data
    if int(kind[0]) == 1:
        _,transform_id,bitrate,maxval,minval,pady,padx,zdim = kind
        padx = int(padx)
        pady = int(pady)
        pseudofile = io.BytesIO(data.tobytes())
        r = imageio.get_reader(pseudofile, format="webm")
        d = np.asarray([it[:(it.shape[0]-pady),:(it.shape[1]-padx),0] for it in r.iter_data()], dtype="float32")
        d = d.swapaxes(int(zdim), 0)
        r.close()
        return _image_decompression_transform(d/255.0*(maxval-minval)+minval, int(transform_id))
    if int(kind[0]) == 2:
        transform_id,quality = kind[1:3]
        lens = np.asarray(kind[3::3]).astype(int)
        maxes = kind[4::3]
        mins = kind[5::3]
        ibase = 0
        ims = []
        for i,l in enumerate(lens):
            pseudofile = io.BytesIO(data[ibase:(ibase+l)].tobytes())
            im = np.asarray(imageio.v3.imread(pseudofile, format_hint=".jpeg"))
            im = _image_decompression_transform(im/255.0*(maxes[i]-mins[i])+mins[i], int(transform_id))
            ims.append(im)
            ibase += l
        return np.asarray(ims, dtype="float32")
    if int(kind[0]) == 3:
        imgfile = zlib.decompress(zlib.decompress(data))
        return np.frombuffer(imgfile, dtype=kind[1]).reshape(*kind[2:].astype('int'))
    raise ValueError(f"Invalid kind {kind}")

def invert_function_numerical(func, point):
    point = np.asarray(point)
    obj = lambda x : np.sum(np.square(point-func(np.asarray([x]))))
    starts = [[0, 0, 0], point]
    opts = []
    for start in starts:
        opts.append(scipy.optimize.minimize(obj, x0=start))
    return min(opts, key=lambda x : x.fun).x

def invert_transform_numerical(tform, points):
    """Perform a numerical inverse transform of 'tform' at 'point'"""
    points = np.asarray(points)
    try:
        if points.ndim == 2:
            return tform.invert().transform(points)[0]
        else:
            return tform.invert().transform(np.asarray([points]))[0]
    except NotImplementedError:
        pass
    if points.ndim == 2:
        return np.asarray([invert_transform_numerical(tform, points[i]) for i in range(0, points.shape[0])])
    return invert_function_numerical(tform.transform, np.asarray([x]))

