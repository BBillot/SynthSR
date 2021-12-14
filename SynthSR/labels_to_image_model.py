"""
If you use this code, please the SynthSR paper in:
https://github.com/BBillot/SynthSR/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers
from ext.neuron import layers as nrn_layers
from ext.lab2im import edit_tensors as et
from ext.lab2im.edit_volumes import get_ras_axes
from ext.lab2im.layers import RandomSpatialDeformation, RandomFlip, MimicAcquisition


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
                          randomise_res=False,
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
    use_real_image = False if output_channel is not None else True
    idx_first_input_channel = np.argmax(input_channels)

    # if only 1 value is given for simulate_registration_error, then replicate for all channels
    simulate_registration_error = utils.reformat_to_list(simulate_registration_error, length=n_channels)

    # reformat resolutions
    # insert dummy slice spacing/thickness for the indices in output_channel, if the corrupted versions of the same
    # channels are not used as inputs. The dummy values won't be used, as synthetic regression targets are not
    # downsampled before being resampled to target resolution. We only insert these dummy values for slice
    # spacing/thickness since an index referring to all channels (input/output) will be used on these two variables.
    labels_shape = utils.reformat_to_list(labels_shape)
    n_dims, _ = utils.get_dims(labels_shape)
    atlas_res = utils.reformat_to_n_channels_array(atlas_res, n_dims, n_channels)
    if output_channel is not None:
        for idx in output_channel:
            if not input_channels[idx]:
                data_res = np.insert(data_res, idx, 1, axis=0)
                thickness = np.insert(thickness, idx, 1, axis=0)
    data_res = atlas_res if data_res is None else utils.reformat_to_n_channels_array(data_res, n_dims, n_channels)
    thickness = data_res if thickness is None else utils.reformat_to_n_channels_array(thickness, n_dims, n_channels)
    downsample = utils.reformat_to_list(downsample, n_channels) if downsample else (np.min(thickness - data_res, 1) < 0)
    atlas_res = atlas_res[0]
    target_res = atlas_res if target_res is None else utils.reformat_to_n_channels_array(target_res, n_dims)[0]
    if isinstance(randomise_res, bool):
        randomise_res = n_channels * [randomise_res]

    # get shapes
    crop_shape, output_shape, padding_margin = get_shapes(labels_shape, output_shape, atlas_res, target_res,
                                                          padding_margin, output_div_by_n)

    # define model inputs
    labels_input = KL.Input(shape=labels_shape + [1], name='labels_input', dtype='int32')
    means_input = KL.Input(shape=list(generation_labels.shape) + [n_channels], name='means_input')
    stds_input = KL.Input(shape=list(generation_labels.shape) + [n_channels], name='std_devs_input')
    list_inputs = [labels_input, means_input, stds_input]

    # add real image to input list if using real regression target
    if use_real_image:
        real_image = KL.Input(shape=labels_shape+[1], dtype='float32', name='real_image_input')
        list_inputs.append(real_image)
    else:
        real_image = None

    # pad labels
    if padding_margin is not None:
        labels = layers.PadAroundCentre(pad_margin=padding_margin)(labels_input)
        labels_shape = labels.get_shape().as_list()[1:n_dims+1]
        if use_real_image:
            real_image = layers.PadAroundCentre(pad_margin=padding_margin)(real_image)
    else:
        labels = labels_input

    # deform labels
    labels._keras_shape = tuple(labels.get_shape().as_list())
    if use_real_image:
        real_image._keras_shape = tuple(real_image.get_shape().as_list())
        labels, real_image = RandomSpatialDeformation(scaling_bounds=scaling_bounds,
                                                      rotation_bounds=rotation_bounds,
                                                      shearing_bounds=shearing_bounds,
                                                      translation_bounds=translation_bounds,
                                                      nonlin_std=nonlin_std,
                                                      nonlin_shape_factor=nonlin_shape_factor,
                                                      inter_method=['nearest', 'linear'])([labels, real_image])
    else:
        labels = RandomSpatialDeformation(scaling_bounds=scaling_bounds,
                                          rotation_bounds=rotation_bounds,
                                          shearing_bounds=shearing_bounds,
                                          translation_bounds=translation_bounds,
                                          nonlin_std=nonlin_std,
                                          nonlin_shape_factor=nonlin_shape_factor,
                                          inter_method='nearest')(labels)

    # cropping
    if crop_shape != labels_shape:
        labels._keras_shape = tuple(labels.get_shape().as_list())
        if use_real_image:
            real_image._keras_shape = tuple(real_image.get_shape().as_list())
            labels, real_image = layers.RandomCrop(crop_shape)([labels, real_image])
        else:
            labels = layers.RandomCrop(crop_shape)(labels)

    # flipping
    if flipping:
        assert aff is not None, 'aff should not be None if flipping is True'
        labels._keras_shape = tuple(labels.get_shape().as_list())
        if use_real_image:
            real_image._keras_shape = tuple(real_image.get_shape().as_list())
            labels, real_image = RandomFlip(get_ras_axes(aff, n_dims)[0], True, generation_labels,
                                            n_neutral_labels)([labels, real_image])
        else:
            labels = RandomFlip(get_ras_axes(aff, n_dims)[0], True, generation_labels, n_neutral_labels)(labels)

    # build synthetic image
    labels._keras_shape = tuple(labels.get_shape().as_list())
    image = layers.SampleConditionalGMM(generation_labels)([labels, means_input, stds_input])

    # give name to output labels
    labels = KL.Lambda(lambda x: x, name='segmentation_target')(labels)

    # loop over synthetic channels
    channels = list()
    targets = list()
    split = KL.Lambda(lambda x: tf.split(x, [1] * n_channels, axis=-1))(image) if (n_channels > 1) else [image]
    for i, channel in enumerate(split):

        # apply bias field
        if input_channels[i]:
            channel._keras_shape = tuple(channel.get_shape().as_list())
            channel = layers.BiasFieldCorruption(bias_field_std, bias_shape_factor, False)(channel)

        # intensity augmentation
        channel._keras_shape = tuple(channel.get_shape().as_list())
        channel = layers.IntensityAugmentation(clip=300, normalise=True, gamma_std=.5)(channel)
        channel._keras_shape = tuple(channel.get_shape().as_list())
        channel = layers.GaussianBlur(sigma=.5)(channel)

        # resample regression target at target resolution if needed
        if not use_real_image:
            if any(c == i for c in output_channel):
                if crop_shape != output_shape:
                    sigma = et.blurring_sigma_for_downsampling(atlas_res, target_res)
                    channel._keras_shape = tuple(channel.get_shape().as_list())
                    channel = layers.GaussianBlur(sigma)(channel)
                    channel = et.resample_tensor(channel, output_shape)
                targets.append(channel)

        # synthetic input channels
        if input_channels[i]:

            # simulate registration error relatively to the first channel (so this does not apply to the first channel)
            if simulate_registration_error[i] & (i != idx_first_input_channel):
                channel._keras_shape = tuple(channel.get_shape().as_list())
                batchsize = KL.Lambda(lambda x: tf.split(tf.shape(x), [1, -1])[0])(channel)
                T = KL.Lambda(lambda x: utils.sample_affine_transform(x, n_dims, rotation_bounds=5,
                                                                      translation_bounds=5))(batchsize)
                Tinv = KL.Lambda(lambda x: tf.linalg.inv(x))(T)
                channel = nrn_layers.SpatialTransformer(interp_method='linear')([channel, T])
            else:
                Tinv = batchsize = None

            channel._keras_shape = tuple(channel.get_shape().as_list())

            # blur and downsample channel
            if randomise_res[i]:
                max_res = np.array([9.] * 3)
                resolution, blur_res = layers.SampleResolution(atlas_res, max_res)(means_input)
                sigma = et.blurring_sigma_for_downsampling(atlas_res, resolution, mult_coef=.42, thickness=blur_res)
                channel = layers.DynamicGaussianBlur(0.75 * max_res / np.array(atlas_res), blur_range)([channel, sigma])
                channel, rel_map = MimicAcquisition(atlas_res, atlas_res, output_shape, True)([channel, resolution])

            else:
                sigma = et.blurring_sigma_for_downsampling(atlas_res, data_res[i], .42, thickness[i])
                channel = layers.GaussianBlur(sigma, blur_range)(channel)
                if downsample[i]:
                    channel, rel_map = et.resample_tensor(channel, output_shape, 'linear', data_res[i], atlas_res, True)
                else:
                    channel, rel_map = et.resample_tensor(channel, output_shape, build_reliability_map=True)

            # align the channels back to the first one with a small error
            if simulate_registration_error[i] & (i != idx_first_input_channel):
                channel._keras_shape = tuple(channel.get_shape().as_list())
                Terr = KL.Lambda(lambda x: utils.sample_affine_transform(x, n_dims, rotation_bounds=.5,
                                                                         translation_bounds=.5))(batchsize)
                Tinv_err = KL.Lambda(lambda x: tf.matmul(x[0], x[1]))([Terr, Tinv])
                channel = nrn_layers.SpatialTransformer(interp_method='linear')([channel, Tinv_err])
                rel_map._keras_shape = tuple(rel_map.get_shape().as_list())
                rel_map = nrn_layers.SpatialTransformer(interp_method='linear')([rel_map, Tinv_err])

            channels.append(channel)
            if build_reliability_maps:
                channels.append(rel_map)

    # concatenate all channels back
    image = KL.Lambda(lambda x: tf.concat(x, -1))(channels) if len(channels) > 1 else channels[0]

    # if no synthetic image is used as regression target, we need to assign the real image to the target!
    if use_real_image:
        real_image._keras_shape = tuple(real_image.get_shape().as_list())
        target = layers.IntensityAugmentation(normalise=True)(real_image)
        if crop_shape != output_shape:
            sigma = et.blurring_sigma_for_downsampling(atlas_res, target_res)
            target._keras_shape = tuple(target.get_shape().as_list())
            target = layers.GaussianBlur(sigma)(target)
            target = et.resample_tensor(target, output_shape)
    else:
        target = KL.Lambda(lambda x: tf.concat(x, axis=-1))(targets) if len(targets) > 1 else targets[0]
    target = KL.Lambda(lambda x: tf.cast(x[0], dtype='float32'), name='regression_target')([target, labels])

    # build model (dummy layer enables to keep the target when plugging this model to other models)
    image = KL.Lambda(lambda x: x[0], name='image_out')([image, target])
    brain_model = Model(inputs=list_inputs, outputs=[image, target])

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
            tmp_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in output_shape]
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
                output_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in output_shape]
                cropping_shape = [int(np.around(output_shape[i] / resample_factor[i], 0)) for i in range(n_dims)]
            # if no resampling, simply check if image_shape is divisible by n
            else:
                cropping_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in labels_shape]
                output_shape = cropping_shape

        # if no need to be divisible by n, simply take cropping_shape as image_shape, and build output_shape
        else:
            cropping_shape = labels_shape
            if resample_factor is not None:
                output_shape = [int(cropping_shape[i] * resample_factor[i]) for i in range(n_dims)]
            else:
                output_shape = cropping_shape

    return cropping_shape, output_shape, padding_margin
