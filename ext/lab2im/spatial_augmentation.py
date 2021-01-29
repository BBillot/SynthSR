# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K

# project imports
from . import utils
from . import edit_volumes

# third-party imports
import ext.neuron.layers as nrn_layers


def deform_tensor(tensor,
                  scaling_bounds=0.15,
                  rotation_bounds=15,
                  enable_90_rotations=False,
                  shearing_bounds=0.012,
                  translation_bounds=False,
                  nonlin_std=2.,
                  nonlin_shape_factor=.0625,
                  inter_method='linear',
                  additional_tensor=None,
                  additional_inter_method='linear'):
    """This function spatially deforms a tensor with a combination of affine and elastic transformations.
    The non linear deformation is obtained by:
    1) a small-size SVF is sampled from a centred normal distribution of random standard deviation.
    2) it is resized with trilinear interpolation to half the shape of the input tensor
    3) it is integrated to obtain a diffeomorphic transformation
    4) finally, it is resized (again with trilinear interpolation) to full image size
    :param tensor: input tensor to deform. Expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    :param scaling_bounds: (optional) range of the random scaling to apply. The scaling factor for each dimension is
    sampled from a uniform distribution of predefined bounds. Can either be:
    1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
    [1-scaling_bounds, 1+scaling_bounds] for each dimension.
    2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
    (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
    3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
     of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
    4) False, in which case scaling is completely turned off.
    Default is scaling_bounds = 0.15 (case 1)
    :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
    and 2, the bounds are centred on 0 rather than 1, i.e. [0+rotation_bounds[i], 0-rotation_bounds[i]].
    Default is rotation_bounds = 15.
    :param enable_90_rotations: (optional) wheter to rotate the input by a random angle chosen in {0, 90, 180, 270}.
    This is done regardless of the value of rotation_bounds. If true, a different value is sampled for each dimension.
    :param shearing_bounds: (optional) same as scaling bounds. Default is shearing_bounds = 0.012.
    :param translation_bounds: (optional) same as scaling bounds. Default is translation_bounds = False, but we
    encourage using it when cropping is deactivated (i.e. when output_shape=None in BrainGenerator).
    :param nonlin_std: (optional) maximum value of the standard deviation of the normal distribution from which we
    sample the small-size SVF. Set to False if you wish to completely turn the elastic deformation off.
    :param nonlin_shape_factor: (optional) if nonlin_std is not False, factor between the shapes of the input tensor
    and the shape of the input non-linear tensor.
    :param inter_method: (optional) interpolation method when deforming the input tensor. Can be 'linear', or 'nearest'
    :param additional_tensor: (optional) in case you want to deform another tensor with the same transformation
    :param additional_inter_method: (optional) interpolation methods for the additional tensor
    :return: tensor of the same shape as volume (a tuple, if additional tensor requested), and if requested, additional
    tensor deformed by the exact same transform.
    """

    apply_affine_trans = (rotation_bounds is not False) | (translation_bounds is not False) | \
                         (scaling_bounds is not False) | (shearing_bounds is not False) | enable_90_rotations
    apply_elastic_trans = (nonlin_std is not False)
    assert (apply_affine_trans is not None) | apply_elastic_trans, 'affine_trans or elastic_trans should be provided'

    # reformat tensor and get its shape
    tensor = KL.Lambda(lambda x: tf.cast(x, dtype='float32'))(tensor)
    tensor._keras_shape = tuple(tensor.get_shape().as_list())
    volume_shape = tensor.get_shape().as_list()[1: -1]
    n_dims = len(volume_shape)
    trans_inputs = list()

    # add affine deformation to inputs list
    if apply_affine_trans:
        affine_trans = sample_affine_transform(rotation_bounds=rotation_bounds,
                                               translation_bounds=translation_bounds,
                                               scaling_bounds=scaling_bounds,
                                               shearing_bounds=shearing_bounds,
                                               n_dims=n_dims,
                                               enable_90_rotations=enable_90_rotations)
        trans_inputs.append(affine_trans)

    # prepare non-linear deformation field and add it to inputs list
    if apply_elastic_trans:

        # sample small field from normal distribution of specified std dev
        small_shape = utils.get_resample_shape(volume_shape, nonlin_shape_factor, n_dims)
        tensor_shape = KL.Lambda(lambda x: tf.shape(x))(tensor)
        split_shape = KL.Lambda(lambda x: tf.split(x, [1, n_dims + 1]))(tensor_shape)
        nonlin_shape = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'), tf.convert_to_tensor(small_shape,
                                 dtype='int32')], axis=0))(split_shape[0])
        nonlin_std_prior = KL.Lambda(lambda x: tf.random.uniform((1, 1), maxval=nonlin_std))([])
        elastic_trans = KL.Lambda(lambda x: tf.random.normal(tf.cast(x[0], 'int32'),
                                                             stddev=x[1]))([nonlin_shape, nonlin_std_prior])
        elastic_trans._keras_shape = tuple(elastic_trans.get_shape().as_list())

        # reshape this field to image size and integrate it
        resize_shape = [max(int(volume_shape[i]/2), small_shape[i]) for i in range(n_dims)]
        nonlin_field = nrn_layers.Resize(size=resize_shape, interp_method='linear')(elastic_trans)
        nonlin_field = nrn_layers.VecInt()(nonlin_field)
        nonlin_field = nrn_layers.Resize(size=volume_shape, interp_method='linear')(nonlin_field)
        trans_inputs.append(nonlin_field)

    # apply deformations and return tensors
    if additional_tensor is None:
        return nrn_layers.SpatialTransformer(interp_method=inter_method)([tensor] + trans_inputs)
    else:
        additional_tensor._keras_shape = tuple(additional_tensor.get_shape().as_list())
        tens1 = nrn_layers.SpatialTransformer(interp_method=inter_method)([tensor] + trans_inputs)
        tens2 = nrn_layers.SpatialTransformer(interp_method=additional_inter_method)([additional_tensor] + trans_inputs)
        return tens1, tens2


def random_cropping(tensor, crop_shape, n_dims=3, additional_tensor=None):
    """Randomly crop an input tensor to a tensor of a given shape. This cropping is applied to all channels.
    :param tensor: input tensor to crop
    :param crop_shape: shape of the cropped tensor, excluding batch and channel dimension.
    :param n_dims: (optional) number of dimensions of the initial image (excluding batch and channel dimensions)
    :param additional_tensor: (optional) in case you want to apply the same cropping to another tensor
    :return: cropped tensor (a tuple, if additional tensor requested)
    example: if tensor has shape [2, 160, 160, 160, 3], and crop_shape=[96, 128, 96], then this function returns a
    tensor of shape [2, 96, 128, 96, 3], with randomly selected cropping indices.
    """

    # get maximum cropping indices in each dimension
    image_shape = tensor.get_shape().as_list()[1:n_dims + 1]
    cropping_max_val = [image_shape[i] - crop_shape[i] for i in range(n_dims)]

    # prepare cropping indices and tensor's new shape (don't crop batch and channel dimensions)
    crop_idx = KL.Lambda(lambda x: tf.zeros([1], dtype='int32'))([])
    for val_idx, val in enumerate(cropping_max_val):  # draw cropping indices for image dimensions
        if val > 0:
            crop_idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'),
                                                      tf.random.uniform([1], 0, val, 'int32')], axis=0))(crop_idx)
        else:
            crop_idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'),
                                                      tf.zeros([1], dtype='int32')], axis=0))(crop_idx)
    crop_idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'),
                                              tf.zeros([1], dtype='int32')], axis=0))(crop_idx)
    patch_shape_tens = KL.Lambda(lambda x: tf.convert_to_tensor([-1] + crop_shape + [-1], dtype='int32'))([])

    # perform cropping
    tensor = KL.Lambda(lambda x: tf.slice(x[0], begin=tf.cast(x[1], dtype='int32'),
                                              size=tf.cast(x[2], dtype='int32')))([tensor, crop_idx, patch_shape_tens])
    if additional_tensor is None:
        return tensor
    else:
        additional_tensor = KL.Lambda(lambda x: tf.slice(x[0], begin=tf.cast(x[1], dtype='int32'), size=tf.cast(x[2],
                                      dtype='int32')))([additional_tensor, crop_idx, patch_shape_tens])
        return tensor, additional_tensor


def label_map_random_flipping(labels, label_list, n_neutral_labels, aff, n_dims=3, flip_rl_only=False,
                              additional_tensor=None):
    """This function flips a label map with a probability of 0.5.
    Right/left label values are also swapped if the label map is flipped in order to preserve the right/left sides.
    :param labels: input label map
    :param label_list: list of all labels contained in labels. Must be ordered as follows, first the neutral labels
    (i.e. non-sided), then left labels and right labels.
    :param n_neutral_labels: number of non-sided labels
    :param aff: affine matrix of the initial input label map, to find the right/left axis.
    :param n_dims: (optional) number of dimensions of the initial image (excluding batch and channel dimensions)
    :params additional_tensor: (optional) in case you want to apply the same flipping to another tensor. This new tensor
    is assumed to be an intensity image, thus it won't undergo any R/L value swapping.
    :return: tensor of the same shape as label map, potentially right/left flipped with correction for sided labels.
    """

    # boolean tensor to decide whether to flip
    rand_flip = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.5))([])

    # swap right and left labels if we later right-left flip the image
    if flip_rl_only:
        n_labels = len(label_list)
        if n_neutral_labels != n_labels:
            rl_split = np.split(label_list, [n_neutral_labels, n_neutral_labels + int((n_labels - n_neutral_labels)/2)])
            flipped_label_list = np.concatenate((rl_split[0], rl_split[2], rl_split[1]))
            labels = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                                  KL.Lambda(lambda x: tf.gather(
                                                      tf.convert_to_tensor(flipped_label_list, dtype='int32'),
                                                      tf.cast(x, dtype='int32')))(y[1]),
                                                  tf.cast(y[1], dtype='int32')))([rand_flip, labels])
        # find right left axis
        ras_axes = edit_volumes.get_ras_axes(aff, n_dims)
        flip_axis_array = [ras_axes[0] + 1]
        flip_axis = KL.Lambda(lambda x: tf.convert_to_tensor(flip_axis_array, dtype='int32'))([])

    # otherwise randomly chose an axis to flip
    else:
        flip_axis = KL.Lambda(lambda x: tf.random.uniform([1], 1, n_dims + 1, dtype='int32'))([])

    # right/left flip
    labels = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                          KL.Lambda(lambda x: K.reverse(x[0], axes=tf.cast(x[1],
                                                                        dtype='int32')))([y[1], y[2]]),
                                          y[1]))([rand_flip, labels, flip_axis])

    if additional_tensor is None:
        return labels
    else:
        additional_tensor = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                                         KL.Lambda(lambda x: K.reverse(x[0], axes=tf.cast(x[1],
                                                                                       dtype='int32')))([y[1], y[2]]),
                                                         y[1]))([rand_flip, additional_tensor, flip_axis])
        return labels, additional_tensor


def restrict_tensor(tensor, axes, boundaries):
    """Reset the edges of a tensor to zero. This is performed only along the given axes.
    The width of the zero-band is randomly drawn from a uniform distribution given in boundaries.
    :param tensor: input tensor
    :param axes: axes along which to reset edges to zero. Can be an int (single axis), or a sequence.
    :param boundaries: numpy array of shape (len(axes), 4). Each row contains the two bounds of the uniform
    distributions from which we draw the width of the zero-bands on each side.
    Those bounds must be expressed in relative side (i.e. between 0 and 1).
    :return: a tensor of the same shape as the input, with bands of zeros along the pecified axes.
    example:
    tensor=tf.constant([[[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]])  # shape = [1,10,10,1]
    axes=1
    boundaries = np.array([[0.2, 0.45, 0.85, 0.9]])

    In this case, we reset the edges along the 2nd dimension (i.e. the 1st dimension after the batch dimension),
    the 1st zero-band will expand from the 1st row to a number drawn from [0.2*tensor.shape[1], 0.45*tensor.shape[1]],
    and the 2nd zero-band will expand from a row drawn from [0.85*tensor.shape[1], 0.9*tensor.shape[1]], to the end of
    the tensor. A possible output could be:
    array([[[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]])  # shape = [1,10,10,1]
    """

    shape = tuple(tensor.get_shape().as_list())
    axes = utils.reformat_to_list(axes, dtype='int')
    boundaries = utils.reformat_to_n_channels_array(boundaries, n_dims=4, n_channels=len(axes))

    # build mask
    mask = KL.Lambda(lambda x: tf.ones_like(x))(tensor)
    for i, axis in enumerate(axes):

        # select restricting indices
        axis_boundaries = boundaries[i, :]
        idx1 = KL.Lambda(lambda x: tf.math.round(tf.random.uniform([1], minval=axis_boundaries[0] * shape[axis],
                                                                   maxval=axis_boundaries[1] * shape[axis])))([])
        idx2 = KL.Lambda(lambda x: tf.math.round(tf.random.uniform([1], minval=axis_boundaries[2] * shape[axis],
                                                                   maxval=axis_boundaries[3] * shape[axis]) - x))(idx1)
        idx3 = KL.Lambda(lambda x: shape[axis] - x[0] - x[1])([idx1, idx2])
        split_idx = KL.Lambda(lambda x: tf.concat([x[0], x[1], x[2]], axis=0))([idx1, idx2, idx3])

        # update mask
        split_list = KL.Lambda(lambda x: tf.split(x[0], tf.cast(x[1], dtype='int32'), axis=axis))([tensor, split_idx])
        tmp_mask = KL.Lambda(lambda x: tf.concat([tf.zeros_like(x[0]), tf.ones_like(x[1]), tf.zeros_like(x[2])],
                                                 axis=axis))([split_list[0], split_list[1], split_list[2]])
        mask = KL.multiply([mask, tmp_mask])

    # mask second_channel
    tensor = KL.multiply([tensor, mask])

    return tensor, mask


def sample_affine_transform(rotation_bounds, translation_bounds, scaling_bounds, shearing_bounds, n_dims,
                            enable_90_rotations=False, return_inv=False):

    if (rotation_bounds is not False) | (enable_90_rotations is not False):
        if n_dims == 2:
            if rotation_bounds is not False:
                rotation = utils.draw_value_from_distribution(rotation_bounds,
                                                              size=1,
                                                              default_range=15.0,
                                                              return_as_tensor=True)
            else:
                rotation = KL.Lambda(lambda x: tf.zeros(1))([])
        else:  # n_dims = 3
            if rotation_bounds is not False:
                rotation = utils.draw_value_from_distribution(rotation_bounds,
                                                              size=n_dims,
                                                              default_range=15.0,
                                                              return_as_tensor=True)
            else:
                rotation = KL.Lambda(lambda x: tf.zeros(3))([])
        if enable_90_rotations:
            rotation = KL.Lambda(lambda x: tf.cast(tf.random.uniform(tf.shape(x), maxval=4, dtype='int32')*90,
                                                   'float32') + x)(rotation)
        T_rot = KL.Lambda(lambda x: create_rotation_transform(x, n_dims))(rotation)
    else:
        T_rot = KL.Lambda(lambda x: tf.eye(3))([])

    if shearing_bounds is not False:
        shearing = utils.draw_value_from_distribution(shearing_bounds,
                                                      size=n_dims ** 2 - n_dims,
                                                      default_range=.01,
                                                      return_as_tensor=True)
        T_shearing = KL.Lambda(lambda x: create_shearing_transform(x))(shearing)
    else:
        T_shearing = KL.Lambda(lambda x: tf.eye(3))([])

    if scaling_bounds is not False:
        scaling = utils.draw_value_from_distribution(scaling_bounds,
                                                     size=n_dims,
                                                     centre=1,
                                                     default_range=.15,
                                                     return_as_tensor=True)
        T_scaling = KL.Lambda(lambda x: tf.linalg.diag(x))(scaling)
    else:
        T_scaling = KL.Lambda(lambda x: tf.eye(3))([])

    T = KL.Lambda(lambda x: tf.matmul(x[2], tf.matmul(x[1], x[0])))([T_rot, T_shearing, T_scaling])

    if translation_bounds is not False:
        translation = utils.draw_value_from_distribution(translation_bounds,
                                                         size=n_dims,
                                                         default_range=5,
                                                         return_as_tensor=True)
        T = KL.Lambda(lambda x: tf.concat([x[0], x[1][:, tf.newaxis]], axis=1))([T, translation])
    else:
        T = KL.Lambda(lambda x: tf.concat([x, tf.zeros([3, 1])], axis=1))(T)

    # build rigid transform and its inverse
    T = KL.Lambda(lambda x: tf.concat([x, tf.constant([0., 0., 0., 1.], shape=[1, 4])], axis=0)[np.newaxis, :])(T)

    if return_inv:
        Tinv = KL.Lambda(lambda x: tf.linalg.inv(x))(T)
        return T, Tinv
    else:
        return T


def create_rotation_transform(rotation, n_dims):
    """build rotation transform from 3d or 2d rotation coefficients. Angles are given in degrees."""
    rotation = rotation * np.pi / 180
    if n_dims == 3:
        Rx_row0 = tf.constant([1, 0, 0], shape=[1, 3], dtype='float32')
        Rx_row1 = tf.stack([tf.zeros(1), tf.cos(rotation[tf.newaxis, 0]), -tf.sin(rotation[tf.newaxis, 0])], axis=1)
        Rx_row2 = tf.stack([tf.zeros(1), tf.sin(rotation[tf.newaxis, 0]), tf.cos(rotation[tf.newaxis, 0])], axis=1)
        Rx = tf.concat([Rx_row0, Rx_row1, Rx_row2], axis=0)

        Ry_row0 = tf.stack([tf.cos(rotation[tf.newaxis, 1]), tf.zeros(1), tf.sin(rotation[tf.newaxis, 1])], axis=1)
        Ry_row1 = tf.constant([0, 1, 0], shape=[1, 3], dtype='float32')
        Ry_row2 = tf.stack([-tf.sin(rotation[tf.newaxis, 1]), tf.zeros(1), tf.cos(rotation[tf.newaxis, 1])], axis=1)
        Ry = tf.concat([Ry_row0, Ry_row1, Ry_row2], axis=0)

        Rz_row0 = tf.stack([tf.cos(rotation[tf.newaxis, 2]), -tf.sin(rotation[tf.newaxis, 2]), tf.zeros(1)], axis=1)
        Rz_row1 = tf.stack([tf.sin(rotation[tf.newaxis, 2]), tf.cos(rotation[tf.newaxis, 2]), tf.zeros(1)], axis=1)
        Rz_row2 = tf.constant([0, 0, 1], shape=[1, 3], dtype='float32')
        Rz = tf.concat([Rz_row0, Rz_row1, Rz_row2], axis=0)

        T_rot = tf.matmul(tf.matmul(Rx, Ry), Rz)

    elif n_dims == 2:
        R_row0 = tf.stack([tf.cos(rotation[tf.newaxis, 0]), tf.sin(rotation[tf.newaxis, 0]), tf.zeros(1)], axis=1)
        R_row2 = tf.stack([-tf.sin(rotation[tf.newaxis, 0]), tf.cos(rotation[tf.newaxis, 0]), tf.zeros(1)], axis=1)
        R_row1 = tf.constant([0, 0, 1], shape=[1, 3], dtype='float32')
        T_rot = tf.concat([R_row0, R_row1, R_row2], axis=0)

    else:
        raise Exception('only supports 2 or 3D.')

    return T_rot


def create_shearing_transform(shearing):
    """build shearing transform from 3d shearing coefficients"""
    shearing_row0 = tf.stack([tf.ones(1), shearing[tf.newaxis, 0], shearing[tf.newaxis, 1]], axis=1)
    shearing_row1 = tf.stack([shearing[tf.newaxis, 2], tf.ones(1), shearing[tf.newaxis, 3]], axis=1)
    shearing_row2 = tf.stack([shearing[tf.newaxis, 4], shearing[tf.newaxis, 5], tf.ones(1)], axis=1)
    return tf.concat([shearing_row0, shearing_row1, shearing_row2], axis=0)
