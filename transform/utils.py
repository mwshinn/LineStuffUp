import imageio
import numpy as np

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
    origin = transform.origin_and_maxpos(im_movable.shape)[0]
    ti = transform.transform_image(im_movable, relative=True)
    new_dims_max = np.ceil(np.max([ti.shape + origin, im_fixed.shape], axis=0)).astype(int)
    new_dims_min = np.floor(np.min([origin, [0,0,0]], axis=0)).astype(int)
    im = np.zeros(new_dims_max-new_dims_min, dtype=float)
    print(new_dims_max, new_dims_min, im.shape)
    blit(im_fixed, im, tuple([0,0,0]-new_dims_min))
    blit(im_movable, im, tuple(origin.astype(int)-new_dims_min))
    return im

def load_image(fn, channel=None):
    img = imageio.imread(fn)
    if channel is None:
        axes = list(np.any((img!=0) & (img!=255), axis=(0,1)))
        return np.mean(img[:,:,axes], axis=2)[None]
    else:
        return img[:,:,channel][None]
