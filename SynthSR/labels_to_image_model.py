# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model

# third-party imports
from ext.lab2im import utils
from ext.lab2im import sample_gmm as l2i_gmm
from ext.lab2im import edit_tensors as l2i_et
from ext.lab2im import spatial_augmentation as l2i_sa
from ext.lab2im import intensity_augmentation as l2i_ia
from ext.neuron import layers as nrn_layers


def labels_to_image_model(labels_shape,
                          input_channels,
                          output_channel,
                          generation_labels,
                          n_neutral_labels,
                          atlas_res,
                          target_res,
                          output_shape=None,
                          output_div_by_n=None,
                          padding_margin=None,
                          flipping=True,
                          aff=None,
                          scaling_bounds=0.15,
                          rotation_bounds=15,
                          shearing_bounds=0.012,
                          translation_bounds=False,
                          nonlin_std=3.,
                          nonlin_shape_factor=.0625,
                          simulate_registration_error=True,
                          data_res=None,
                          thickness=None,
                          downsample=False,
                          build_reliability_maps=False,
                          blur_range=1.15,
                          bias_field_std=.3,
                          bias_shape_factor=.025):
    """
    This is used for imputation (and possibly synthesis)
    - the target is a crisp image
    - some channels may only be inputs, and some only targets
    - the target can be a separate real scan
    - it produces additional volumes that tell whether a modality is measured or interpolated at every location
    - models the fact that registration may be needed to bring images into alignment (i.e., acquisitions are not
      perfectly parallel / ortoghonal)
    """

    # vector indicating which synthetic channels will be used as inputs to the UNet
    n_channels = len(input_channels)
    if output_channel is not None:
        use_real_image = False
    else:
        use_real_image = True
    idx_first_input_channel = np.argmax(input_channels)
    n_input_channels = n_channels - np.sum(np.logical_not(input_channels))

    # if only 1 value is given for  simulate_registration_error, then replicate for all channels
    if type(simulate_registration_error)==bool:
        simulate_registration_error = [simulate_registration_error] * n_channels

    # reformat resolutions
    labels_shape = utils.reformat_to_list(labels_shape)
    n_dims, _ = utils.get_dims(labels_shape)
    atlas_res = utils.reformat_to_n_channels_array(atlas_res, n_dims=n_dims, n_channels=n_channels)
    if data_res is None:  # data_res assumed to be the same as the atlas
        data_res = atlas_res
    else:
        data_res = utils.reformat_to_n_channels_array(data_res, n_dims=n_dims, n_channels=n_input_channels)
    atlas_res = atlas_res[0]
    if target_res is None:
        target_res = atlas_res
    else:
        target_res = utils.reformat_to_n_channels_array(target_res, n_dims)[0]
    thickness = utils.reformat_to_n_channels_array(thickness, n_dims=n_dims, n_channels=n_input_channels)

    # Eugenio removed this: output channels are normal synthetic channels...
    # # insert dummy slice spacing/thickness for output_channel (they won't be used per se as synthetic regression targets
    # # are not downsampled) because an index referring to all channels (input/output) will be used on these two variables
    # if output_channel is not None:
    #     if not input_channels[output_channel]:
    #         data_res = np.insert(data_res, output_channel, 1, axis=0)
    #         thickness = np.insert(thickness, output_channel, 1, axis=0)

    # get shapes
    crop_shape, output_shape, padding_margin = get_shapes(labels_shape, output_shape, atlas_res, target_res,
                                                          padding_margin, output_div_by_n)

    # create new_label_list and corresponding LUT to make sure that labels go from 0 to N-1
    n_generation_labels = generation_labels.shape[0]
    new_generation_label_list, lut = utils.rearrange_label_list(generation_labels)

    # define model inputs
    labels_input = KL.Input(shape=labels_shape+[1], name='labels_input')
    means_input = KL.Input(shape=list(new_generation_label_list.shape) + [n_channels], name='means_input')
    stds_input = KL.Input(shape=list(new_generation_label_list.shape) + [n_channels], name='stds_input')
    list_inputs = [labels_input, means_input, stds_input]

    # add real image to input list if using real regression target
    if use_real_image:
        real_image_input = KL.Input(shape=labels_shape+[1], name='real_image_input')
        list_inputs.append(real_image_input)
        real_image = KL.Lambda(lambda x: tf.cast(x, dtype='float32'))(real_image_input)
    else:
        real_image = None

    # convert labels to new_label_list
    labels = l2i_et.convert_labels(labels_input, lut)

    # pad labels
    if padding_margin is not None:
        pad = np.transpose(np.array([[0] + padding_margin + [0]] * 2))
        labels = KL.Lambda(lambda x: tf.pad(x, tf.cast(tf.convert_to_tensor(pad), dtype='int32')), name='pad')(labels)
        labels_shape = labels.get_shape().as_list()[1:n_dims+1]
        if use_real_image:
            real_image = KL.Lambda(lambda x: tf.pad(x, tf.cast(tf.convert_to_tensor(pad), dtype='int32')),
                                   name='pad_real_image')(real_image)

    # deform labels
    if (scaling_bounds is not False) | (rotation_bounds is not False) | (shearing_bounds is not False) | \
       (translation_bounds is not False) | (nonlin_std is not False):
        if use_real_image:
            labels, real_image = l2i_sa.deform_tensor(labels,
                                                      scaling_bounds=scaling_bounds,
                                                      rotation_bounds=rotation_bounds,
                                                      shearing_bounds=shearing_bounds,
                                                      translation_bounds=translation_bounds,
                                                      nonlin_std=nonlin_std,
                                                      nonlin_shape_factor=nonlin_shape_factor,
                                                      inter_method='nearest',
                                                      additional_tensor=real_image)
        else:
            labels = l2i_sa.deform_tensor(labels,
                                          scaling_bounds=scaling_bounds,
                                          rotation_bounds=rotation_bounds,
                                          shearing_bounds=shearing_bounds,
                                          translation_bounds=translation_bounds,
                                          nonlin_std=nonlin_std,
                                          nonlin_shape_factor=nonlin_shape_factor,
                                          inter_method='nearest')
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='int32'))(labels)

    # cropping
    if crop_shape != labels_shape:
        if use_real_image:
            labels, real_image = l2i_sa.random_cropping(labels, crop_shape, n_dims, real_image)
        else:
            labels = l2i_sa.random_cropping(labels, crop_shape, n_dims)

    # flipping
    if flipping:
        assert aff is not None, 'aff should not be None if flipping is True'
        if use_real_image:
            labels, real_image = l2i_sa.label_map_random_flipping(labels, new_generation_label_list, n_neutral_labels,
                                                                  aff, n_dims, True, real_image)
        else:
            labels = l2i_sa.label_map_random_flipping(labels, new_generation_label_list, n_neutral_labels, aff, n_dims,
                                                      flip_rl_only=True)

    # build synthetic image
    image = l2i_gmm.sample_gmm_conditioned_on_labels(labels, means_input, stds_input, n_generation_labels, n_channels)

    # split synthetic channels
    if n_channels > 1:
        split = KL.Lambda(lambda x: tf.split(x, [1] * n_channels, axis=-1))(image)
    else:
        split = [image]
    mask = KL.Lambda(lambda x: tf.where(tf.greater(x, 0), tf.ones_like(x, dtype='float32'),
                                        tf.zeros_like(x, dtype='float32')))(labels)

    # loop over synthetic channels
    processed_channels = list()
    regression_target = list()
    for i, channel in enumerate(split):

        # apply bias field
        if (bias_field_std is not False) & input_channels[i]:
            channel = l2i_ia.bias_field_augmentation(channel, bias_field_std, bias_shape_factor)

        # intensity augmentation
        channel = KL.Lambda(lambda x: K.clip(x, 0, 300))(channel)
        channel = l2i_ia.min_max_normalisation(channel)
        channel = l2i_ia.gamma_augmentation(channel, std=0.5)
        kernels_list_cosmetic = l2i_et.get_gaussian_1d_kernels([.5] * 3)
        channel = l2i_et.blur_channel(channel, mask, kernels_list_cosmetic, n_dims, True)

        # synthetic regression target
        if any(c==i for c in output_channel):
            target = KL.Lambda(lambda x: tf.cast(x, dtype='float32'))(channel)
            # resample regression target at target resolution if needed
            if crop_shape != output_shape:
                sigma = utils.get_std_blurring_mask_for_downsampling(target_res, atlas_res)
                kernels_list = l2i_et.get_gaussian_1d_kernels(sigma)
                target = l2i_et.blur_tensor(target, kernels_list, n_dims=n_dims)
                target = l2i_et.resample_tensor(target, output_shape)
            regression_target.append(target)

        # synthetic input channels
        if input_channels[i]:

            # simulate registration error relatively to the first channel (so this does not apply to the first channel)
            if simulate_registration_error[i] & (i != idx_first_input_channel):
                T, Tinv = l2i_sa.sample_affine_transform(5, 5, False, False, n_dims=n_dims, return_inv=True)
                channel._keras_shape = tuple(channel.get_shape().as_list())
                channel = nrn_layers.SpatialTransformer(interp_method='linear')([channel, T])
            else:
                Tinv = None

            # blur channel
            sigma = utils.get_std_blurring_mask_for_downsampling(data_res[i], atlas_res, thickness[i], mult_coef=0.42)
            kernels_list = l2i_et.get_gaussian_1d_kernels(sigma, blurring_range=blur_range)
            channel = l2i_et.blur_channel(channel, mask, kernels_list, n_dims)

            # resample channel
            if downsample:  # downsample if requested
                channel, rel_map = l2i_et.resample_tensor(channel, output_shape, 'linear', data_res[i], atlas_res,
                                                          build_reliability_map=True)
            elif thickness[i] is not None:  # automatically downsample if data_res > thickness
                diff = [thickness[i][dim_idx] - data_res[i][dim_idx] for dim_idx in range(n_dims)]
                if min(diff) < 0:
                    channel, rel_map = l2i_et.resample_tensor(channel, output_shape, 'linear', data_res[i], atlas_res,
                                                              build_reliability_map=True)
                else:
                    channel, rel_map = l2i_et.resample_tensor(channel, output_shape, 'linear',
                                                              build_reliability_map=True)
            else:
                channel, rel_map = l2i_et.resample_tensor(channel, output_shape, 'linear', build_reliability_map=True)

            # align the channels back to the first one with a small error
            if simulate_registration_error[i] & (i != idx_first_input_channel):
                Terr = l2i_sa.sample_affine_transform(.5, .5, False, False, n_dims=n_dims, return_inv=False)
                Tinv_err = KL.Lambda(lambda x: tf.matmul(x[0], x[1]))([Terr, Tinv])
                channel._keras_shape = tuple(channel.get_shape().as_list())
                channel = nrn_layers.SpatialTransformer(interp_method='linear')([channel, Tinv_err])
                rel_map = KL.Lambda(lambda x: tf.cast(x, dtype='float32'))(rel_map)
                rel_map._keras_shape = tuple(rel_map.get_shape().as_list())
                rel_map = nrn_layers.SpatialTransformer(interp_method='linear')([rel_map, Tinv_err])

            processed_channels.append(channel)
            if build_reliability_maps:
                processed_channels.append(rel_map)

    # concatenate channels back
    if len(processed_channels) > 1:
        image = KL.Lambda(lambda x: tf.concat(x, axis=-1))(processed_channels)
    else:
        image = processed_channels[0]

    # if no synthetic image is used as regression target, we need to assign the real image to the target!
    if use_real_image:
        real_image = l2i_ia.min_max_normalisation(real_image)
        final_target = KL.Lambda(lambda x: tf.cast(x, dtype='float32'), name='regression_target')(real_image)
    else:
        if len(regression_target) > 1:
            final_target = KL.Lambda(lambda x: tf.concat(x, axis=-1), name='regression_target')(regression_target)
        else:
            final_target = KL.Lambda(lambda x: tf.cast(x, dtype='float32'), name='regression_target')(regression_target[0])

    # build model (dummy layer enables to keep the target when plugging this model to other models)
    image = KL.Lambda(lambda x: x[0], name='image_out')([image, final_target])
    brain_model = Model(inputs=list_inputs, outputs=[image, final_target])

    return brain_model


def get_shapes(labels_shape, output_shape, atlas_res, target_res, padding_margin, output_div_by_n):

    # reformat resolutions to lists
    atlas_res = utils.reformat_to_list(atlas_res)
    n_dims = len(atlas_res)
    target_res = utils.reformat_to_list(target_res)

    # get new labels shape if padding
    if padding_margin is not None:
        padding_margin = utils.reformat_to_list(padding_margin, length=n_dims, dtype='int')
        labels_shape = [labels_shape[i] + 2 * padding_margin[i] for i in range(n_dims)]

    # get resampling factor
    if atlas_res != target_res:
        resample_factor = [atlas_res[i] / float(target_res[i]) for i in range(n_dims)]
    else:
        resample_factor = None

    # output shape specified, need to get cropping shape, and resample shape if necessary
    if output_shape is not None:
        output_shape = utils.reformat_to_list(output_shape, length=n_dims, dtype='int')

        # make sure that output shape is smaller or equal to label shape
        if resample_factor is not None:
            output_shape = [min(int(labels_shape[i] * resample_factor[i]), output_shape[i]) for i in range(n_dims)]
        else:
            output_shape = [min(labels_shape[i], output_shape[i]) for i in range(n_dims)]

        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            tmp_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, smaller_ans=True)
                         for s in output_shape]
            if output_shape != tmp_shape:
                print('output shape {0} not divisible by {1}, changed to {2}'.format(output_shape, output_div_by_n,
                                                                                     tmp_shape))
                output_shape = tmp_shape

        # get cropping and resample shape
        if resample_factor is not None:
            cropping_shape = [int(np.around(output_shape[i]/resample_factor[i], 0)) for i in range(n_dims)]
        else:
            cropping_shape = output_shape

    # no output shape specified, so no cropping unless label_shape is not divisible by output_div_by_n
    else:

        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:

            # if resampling, get the potential output_shape and check if it is divisible by n
            if resample_factor is not None:
                output_shape = [int(labels_shape[i] * resample_factor[i]) for i in range(n_dims)]
                output_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, smaller_ans=True)
                                for s in output_shape]
                cropping_shape = [int(np.around(output_shape[i] / resample_factor[i], 0)) for i in range(n_dims)]
            # if no resampling, simply check if image_shape is divisible by n
            else:
                cropping_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, smaller_ans=True)
                                  for s in labels_shape]
                output_shape = cropping_shape

        # if no need to be divisible by n, simply take cropping_shape as image_shape, and build output_shape
        else:
            cropping_shape = labels_shape
            if resample_factor is not None:
                output_shape = [int(cropping_shape[i] * resample_factor[i]) for i in range(n_dims)]
            else:
                output_shape = cropping_shape

    return cropping_shape, output_shape, padding_margin
