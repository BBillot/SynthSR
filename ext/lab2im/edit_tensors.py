"""This file contains functions to edit keras/tensorflow tensors.
A lot of them are used in lab2im_model, and we provide them here separately, so they can be re-used easily.
The functions are classified in three categories:
1- blurring functions: They contain functions to create blurring tensors and to apply the obtained kernels:
    -get_std_blurring_mask_for_downsampling
    -blur_tensor
    -get_gaussian_1d_kernels
    -sample_random_resolution
    -blur_channel
2- resampling function: function to resample a tensor to a specified resolution.
    -resample_tensor
3- converting label values: these functions only apply to tensors with a limited set of integers as values (typically
label map tensors). It contains:
    -convert_labels
    -reset_label_values_to_zero
4- padding tensor
    -pad_tensor
"""

# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
import tensorflow_probability as tfp

# project imports
from . import utils

# third-party imports
import ext.neuron.layers as nrn_layers


# ------------------------------------------------- blurring functions -------------------------------------------------

def get_std_blurring_mask_for_downsampling(downsample_res, current_res, thickness=None):
    """Compute standard deviations of 1d gaussian masks for image blurring before downsampling.
    :param downsample_res: resolution to downsample to. Can be a 1d numpy array or list, or a tensor.
    :param current_res: resolution of the volume before downsampling.
    Can be a 1d numpy array or list or tensor of the same length as downsample res.
    :param thickness: (optional) slices thickness in each dimension.
    Can be a 1d numpy array or list of the same length as downsample res.
    :return: standard deviation of the blurring masks given as as the same type as downsample_res (list or tensor).
    """

    n_dims = len(current_res)

    if tf.is_tensor(downsample_res):

        if thickness is not None:
            tmp_down_res = KL.Lambda(lambda x: tf.math.minimum(tf.convert_to_tensor(thickness, dtype='float32'),
                                                                     x))(downsample_res)
        else:
            tmp_down_res = downsample_res

        current_res = KL.Lambda(lambda x: tf.convert_to_tensor(current_res, dtype='float32'))([])
        sigma = KL.Lambda(lambda x:
                          tf.where(tf.math.equal(x[0], x[1]), 0.5, 0.75 * x[0] / x[1]))([tmp_down_res, current_res])
        sigma = KL.Lambda(lambda x: tf.where(tf.math.equal(x[0], 0.), 0., x[1]))([tmp_down_res, sigma])

    else:

        # reformat data resolution at which we blur
        if thickness is not None:
            downsample_res = [min(downsample_res[i], thickness[i]) for i in range(n_dims)]

        # build 1d blurring kernels for each direction
        sigma = [0] * n_dims
        for i in range(n_dims):
            # define sigma
            if downsample_res[i] == 0:
                sigma[i] = 0
            elif current_res[i] == downsample_res[i]:
                sigma[i] = np.float32(0.5)
            else:
                sigma[i] = np.float32(0.75 * np.around(downsample_res[i] / current_res[i], 3))

    return sigma


def blur_tensor(tensor, list_kernels, n_dims=3):
    """Blur image with masks in list_kernels, if they are not None or do not contain NaN values."""
    for kernel in list_kernels:
        if kernel is not None:
            tensor = KL.Lambda(lambda x: K.switch(tf.math.reduce_any(tf.math.is_nan(x[1])),
                                                  x[0],
                                                  tf.nn.convolution(x[0], x[1], padding='SAME',
                                                                    strides=[1]*n_dims)))([tensor, kernel])
    return tensor


def get_gaussian_1d_kernels(sigma, max_sigma=None, blurring_range=None):
    """This function builds a list of 1d gaussian blurring kernels.
    The produced tensors are designed to be used with tf.nn.convolution.
    The number of dimensions of the image to blur is assumed to be the length of sigma.
    :param sigma: std deviation of the gaussian kernels to build. Must be a sequence of size (n_dims,)
    (excluding batch and channel dimensions). This can also be provided as a tensor (of size (n_dims,)) to be able to
    use different values for sigma in a keras model.
    :param max_sigma: (optional) maximum possible value for sigma (when it varies, e.g. when provided as a tensor).
    This is used to compute the size of the returned kernels. It *must* be provided when sigma is a tensor, but is
    optional otherwise. Must be a numpy array, of the same size as sigma.
    :param blurring_range: (optional) if not None, this introduces a randomness in the blurring kernels,
    where sigma is now multiplied by a coefficient dynamically sampled from a uniform distribution with bounds
    [1/blurring_range, blurring_range].
    :return: a list of 1d blurring kernels
    """

    # convert sigma into a tensor
    if not tf.is_tensor(sigma):
        sigma_tens = KL.Lambda(lambda x: tf.convert_to_tensor(utils.reformat_to_list(sigma), dtype='float32'))([])
    else:
        assert max_sigma is not None, 'max_sigma must be provided when sigma is given as a tensor'
        sigma_tens = sigma
    n_dims = sigma_tens.get_shape().as_list()[0]
    list_sigma = KL.Lambda(lambda x: tf.split(x, [1] * n_dims, axis=0))(sigma_tens)

    # reset blurring range to 1
    if blurring_range is None:
        blurring_range = 1.

    # reformat sigma and get size of blurring kernels
    if max_sigma is None:
        max_sigma = np.array(sigma) * blurring_range  # np.array(sigma) is defined as sigma is not a tensor in this case
    else:
        max_sigma = np.array(utils.reformat_to_list(max_sigma, length=n_dims))
    size = np.int32(np.ceil(2.5 * max_sigma) / 2)

    kernels_list = list()
    for i in range(n_dims):

        # build 1d kernel
        random_coef = KL.Lambda(lambda x: tf.random.uniform((1,), 1 / blurring_range, blurring_range))([])
        kernel = KL.Lambda(lambda x: tfp.distributions.Normal(0., x[0] * x[1]).prob(tf.range(start=-size[i],
                           limit=size[i] + 1, dtype=tf.float32)))([random_coef, list_sigma[i]])
        kernel = KL.Lambda(lambda x: x / tf.reduce_sum(x))(kernel)

        # add dimensions
        for j in range(n_dims):
            if j < i:
                kernel = KL.Lambda(lambda x: tf.expand_dims(x, 0))(kernel)
            elif j > i:
                kernel = KL.Lambda(lambda x: tf.expand_dims(x, -1))(kernel)
        kernels_list.append(KL.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, -1), -1))(kernel))

    return kernels_list


def sample_random_resolution(min_resolution, max_resolution):
    """This function returns a resolution tensor where all values are identical to the values in min_resolution,
    except for a single random axis, for which the value is sampled uniformly between the values of the corresponding
    axis of min_resolution and max_resolution.
    :param min_resolution: a numpy array with minimum resolution for each axis
    :param max_resolution: a numpy array (same size as min_resolution) with maximum resolution for each axis.
    :return: a tensor of shape (len(min_resolution),)
    examples: if min_resolution = np.array([1. ,1. 1.]) and max_resolution = np.array([5. ,8. 2.]), possible sampled
    values can be: tf.Tensor([1., 7.5, 1.]), tf.Tensor([1.2., 1., 1.]), tf.Tensor([1., 1., 1.9])
    """

    # check dimension
    assert len(min_resolution) == len(max_resolution), \
        'min and max resolution must have the same length, had {0} and {1}'.format(min_resolution, max_resolution)
    n_dims = len(min_resolution)

    # initialise random resolution with minimum resolution
    resolution = KL.Lambda(lambda x: tf.convert_to_tensor(min_resolution, dtype='float32'))([])
    new_resolution = KL.Lambda(lambda x:
                               tf.random.uniform(shape=(n_dims,), minval=min_resolution, maxval=max_resolution))([])

    # select dimension
    dim = KL.Lambda(lambda x: tf.random.uniform(shape=(1, 1), minval=0, maxval=n_dims, dtype='int32'))([])
    mask = KL.Lambda(lambda x: tf.zeros((3,), dtype='bool'))([])
    mask = KL.Lambda(lambda x:
                     tf.tensor_scatter_nd_update(x[0], x[1], tf.convert_to_tensor([True], dtype='bool')))([mask, dim])

    # replace resolution in selected axis
    new_res = KL.Lambda(lambda x: tf.where(x[0], x[1], x[2]))([mask, new_resolution, resolution])

    return new_res


def blur_channel(tensor, mask, kernels_list, n_dims, blur_background=True):
    """Blur a tensor with a list of kernels.
    If blur_background is True, this function enforces a zero background after blurring in 20% of the cases.
    If blur_background is False, this function corrects edge-blurring effects and replaces the zero-backgound by a low
    intensity gaussian noise.
    :param tensor: a input tensor
    :param mask: mask of non-background regions in the input tensor
    :param kernels_list: list of blurring 1d kernels
    :param n_dims: number of dimensions of the initial image (excluding batch and channel dimensions)
    :param blur_background: (optional) whether to correct for edge-blurring effects
    :return: blurred tensor with background augmentation
    """

    # blur image
    tensor = blur_tensor(tensor, kernels_list, n_dims)

    if blur_background:  # background already blurred with the rest of the image

        # enforce zero background in 20% of the cases
        rand = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.8))([])
        tensor = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                              KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'),
                                                                           x[0], tf.zeros_like(x[0])))([y[1], y[2]]),
                                              y[1]))([rand, tensor, mask])

    else:  # correct for edge blurring effects

        # blur mask and correct edge blurring effects
        blurred_mask = blur_tensor(mask, kernels_list, n_dims)
        tensor = KL.Lambda(lambda x: x[0] / (x[1] + K.epsilon()))([tensor, blurred_mask])

        # replace zero background by low intensity background in 50% of the cases
        rand = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.5))([])
        bckgd_mean = KL.Lambda(lambda x: tf.random.uniform((1, 1), 0, 20))([])
        bckgd_std = KL.Lambda(lambda x: tf.random.uniform((1, 1), 0, 10))([])
        bckgd_mean = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                                  KL.Lambda(lambda x: tf.zeros_like(x))(y[1]),
                                                  y[1]))([rand, bckgd_mean])
        bckgd_std = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                                 KL.Lambda(lambda x: tf.zeros_like(x))(y[1]),
                                                 y[1]))([rand, bckgd_std])
        background = KL.Lambda(lambda x: x[1] + x[2]*tf.random.normal(tf.shape(x[0])))([tensor, bckgd_mean, bckgd_std])
        background_kernels = get_gaussian_1d_kernels(sigma=[1] * 3)
        background = blur_tensor(background, background_kernels, n_dims)
        tensor = KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0], x[2]))([tensor, mask, background])

    return tensor


# ------------------------------------------------ resampling functions ------------------------------------------------

def resample_tensor(tensor,
                    resample_shape,
                    interp_method='linear',
                    subsample_res=None,
                    volume_res=None,
                    build_reliability_map=False):
    """This function resamples a volume to resample_shape. It does not apply any pre-filtering.
    A prior downsampling step can be added if subsample_res is specified. In this case, volume_res should also be
    specified, in order to calculate the downsampling ratio. A reliability map can also be returned to indicate which
    slices were interpolated during resampling from the downsampled to final tensor.
    :param tensor: tensor
    :param resample_shape: list or numpy array of size (n_dims,)
    :param interp_method: (optional) interpolation method for resampling, 'linear' (default) or 'nearest'
    :param subsample_res: (optional) if not None, this triggers a downsampling of the volume, prior to the resampling
    step. List or numpy array of size (n_dims,). Default si None.
    :param volume_res: (optional) if subsample_res is not None, this should be provided to compute downsampling ratio.
    list or numpy array of size (n_dims,). Default is None.
    :param build_reliability_map: whether to return reliability map along with the resampled tensor. This map indicates
    which slices of the resampled tensor are interpolated (0=interpolated, 1=real slice, in between=degree of realness).
    :return: resampled volume, with reliability map if necessary.
    """

    # reformat resolutions to lists
    subsample_res = utils.reformat_to_list(subsample_res)
    volume_res = utils.reformat_to_list(volume_res)
    n_dims = len(resample_shape)

    # downsample image
    tensor_shape = tensor.get_shape().as_list()[1:-1]
    downsample_shape = tensor_shape  # will be modified if we actually downsample

    if subsample_res is not None:
        assert volume_res is not None, 'volume_res must be given when providing a subsampling resolution.'
        assert len(subsample_res) == len(volume_res), 'subsample_res and volume_res must have the same length, ' \
                                                      'had {0}, and {1}'.format(len(subsample_res), len(volume_res))
        if subsample_res != volume_res:

            # get shape at which we downsample
            downsample_shape = [int(tensor_shape[i] * volume_res[i] / subsample_res[i]) for i in range(n_dims)]

            # downsample volume
            tensor._keras_shape = tuple(tensor.get_shape().as_list())
            tensor = nrn_layers.Resize(size=downsample_shape, interp_method='nearest')(tensor)

    # resample image at target resolution
    if resample_shape != downsample_shape:  # if we didn't dowmsample downsample_shape = tensor_shape
        tensor._keras_shape = tuple(tensor.get_shape().as_list())
        tensor = nrn_layers.Resize(size=resample_shape, interp_method=interp_method)(tensor)

    # compute reliability maps if necessary and return results
    if build_reliability_map:

        # compute maps only if we downsampled
        if downsample_shape != tensor_shape:

            # compute upsampling factors
            upsampling_factors = np.array(resample_shape) / np.array(downsample_shape)

            # build reliability map
            reliability_map = 1
            for i in range(n_dims):
                loc_float = np.arange(0, resample_shape[i], upsampling_factors[i])
                loc_floor = np.int32(np.floor(loc_float))
                loc_ceil = np.int32(np.clip(loc_floor + 1, 0, resample_shape[i] - 1))
                tmp_reliability_map = np.zeros(resample_shape[i])
                tmp_reliability_map[loc_floor] = 1 - (loc_float - loc_floor)
                tmp_reliability_map[loc_ceil] = tmp_reliability_map[loc_ceil] + (loc_float - loc_floor)
                shape = [1, 1, 1]
                shape[i] = resample_shape[i]
                reliability_map = reliability_map * np.reshape(tmp_reliability_map, shape)
            shape = KL.Lambda(lambda x: tf.shape(x))(tensor)
            mask = KL.Lambda(lambda x: tf.reshape(tf.convert_to_tensor(reliability_map, dtype='float32'),
                                                  shape=x))(shape)

        # otherwise just return an all-one tensor
        else:
            mask = KL.Lambda(lambda x: tf.ones_like(x))(tensor)

        return tensor, mask

    else:
        return tensor


# ------------------------------------------------ convert label values ------------------------------------------------

def convert_labels(label_map, labels_list):
    """Change all labels in label_map by the values in labels_list"""
    return KL.Lambda(lambda x: tf.gather(tf.convert_to_tensor(labels_list, dtype='int32'),
                                         tf.cast(x, dtype='int32')))(label_map)


def reset_label_values_to_zero(label_map, labels_to_reset):
    """Reset to zero all occurences in label_map of the values contained in labels_to_remove.
    :param label_map: tensor
    :param labels_to_reset: list of values to reset to zero
    """
    for lab in labels_to_reset:
        label_map = KL.Lambda(lambda x: tf.where(tf.equal(tf.cast(x, dtype='int32'),
                                                          tf.cast(tf.convert_to_tensor(lab), dtype='int32')),
                                                 tf.zeros_like(x, dtype='int32'),
                                                 tf.cast(x, dtype='int32')))(label_map)
    return label_map


# ---------------------------------------------------- pad tensors -----------------------------------------------------

def pad_tensor(tensor, padding_shape, pad_value=0):
    """Pad tensor, around its centre, to specified shape.
    :param tensor: tensor to pad
    :param padding_shape: shape of the returned padded tensor. Can be a list or a numy 1d array, of the same length as
    the numbe of dimensions of the tensor (including batch and channel dimensions).
    :param pad_value: (optional) value by which to pad the tensor. Default is 0.
    """

    # get shapes and padding margins
    tensor_shape = KL.Lambda(lambda x: tf.shape(x))(tensor)
    padding_shape = KL.Lambda(lambda x: tf.math.maximum(tf.cast(x, dtype='int32'),
                              tf.convert_to_tensor(padding_shape, dtype='int32')))(tensor_shape)

    # build padding margins
    min_margins = KL.Lambda(lambda x: tf.cast((x[0] - x[1]) / 2, dtype='int32'))([padding_shape, tensor_shape])
    max_margins = KL.Lambda(lambda x: tf.cast((x[0] - x[1]) - x[2], dtype='int32'))([padding_shape, tensor_shape,
                                                                                     min_margins])
    margins = KL.Lambda(lambda x: tf.stack([tf.cast(x[0], dtype='int32'),
                                            tf.cast(x[1], dtype='int32')], axis=-1))([min_margins, max_margins])

    # pad tensor
    padded_tensor = KL.Lambda(lambda x: tf.pad(x[0], tf.cast(x[1], dtype='int32'), mode='CONSTANT',
                                               constant_values=pad_value))([tensor, margins])
    return padded_tensor
