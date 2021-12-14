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
import keras.backend as K
from keras.models import Model

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers


def metrics_model(input_model, loss_cropping=16, metrics='l1', work_with_residual_channel=None):

    # If probabilistic, split predictions of intensities and spreads
    if metrics == 'laplace':
        n_channels = int(input_model.outputs[0].shape[-1]/2)
        intensities_list = list()
        spreads_list = list()
        tensor = input_model.outputs[0]
        for c in range(n_channels):
            tmp_intensities = KL.Lambda(lambda x: tf.expand_dims(x[..., c], axis=-1))(tensor)
            intensities_list.append(tmp_intensities)
            tmp_spreads = KL.Lambda(lambda x: tf.expand_dims(x[..., c + n_channels], axis=-1))(tensor)
            spreads_list.append(tmp_spreads)
        if n_channels > 1:
            intensities_tensor = KL.Lambda(lambda x: tf.concat(x, axis=-1))(intensities_list)
            spreads_tensor = KL.Lambda(lambda x: tf.concat(x, axis=-1))(spreads_list)
        else:
            intensities_tensor = intensities_list[0]
            spreads_tensor = spreads_list[0]
    else:
        intensities_tensor = input_model.outputs[0]
        spreads_tensor = None

    # add residual if needed
    if work_with_residual_channel is None:
        intensities_tensor = KL.Lambda(lambda x: x, name='predicted_image')(intensities_tensor)
    else:
        slice_list = list()
        for c in work_with_residual_channel:
            tensor = input_model.get_layer('image_out').output
            tmp_slice = KL.Lambda(lambda x: tf.expand_dims(x[..., c], axis=-1))(tensor)
            slice_list.append(tmp_slice)
        if len(slice_list) > 1:
            slices = KL.Lambda(lambda x: tf.concat(x, axis=-1))(slice_list)
        else:
            slices = slice_list[0]
        intensities_tensor = KL.Add(name='predicted_image')([slices, intensities_tensor])

    # get crisp, ground truth image
    image_gt = input_model.get_layer('regression_target').output
    image_gt = KL.Lambda(lambda x: x, name='target')(image_gt)

    # crop output to evaluate loss function in centre patch
    if loss_cropping is not None:
        # format loss_cropping
        target_shape = image_gt.get_shape().as_list()[1:-1]
        n_dims, _ = utils.get_dims(target_shape)
        loss_cropping = utils.reformat_to_list(loss_cropping, length=n_dims)

        # perform cropping
        begin_idx = [int((target_shape[i] - loss_cropping[i]) / 2) for i in range(n_dims)]
        image_gt = KL.Lambda(lambda x: tf.slice(x, begin=tf.convert_to_tensor([0] + begin_idx + [0], dtype='int32'),
                                                size=tf.convert_to_tensor([-1] + loss_cropping + [-1], dtype='int32')),
                             name='cropping_gt')(image_gt)
        intensities_tensor = KL.Lambda(lambda x:
                                       tf.slice(x, begin=tf.convert_to_tensor([0] + begin_idx + [0], dtype='int32'),
                                                size=tf.convert_to_tensor([-1] + loss_cropping + [-1], dtype='int32')),
                                       name='cropping_pred')(intensities_tensor)
        if metrics == 'laplace':
            spreads_tensor = KL.Lambda(lambda x:
                                       tf.slice(x, begin=tf.convert_to_tensor([0] + begin_idx + [0], dtype='int32'),
                                                size=tf.convert_to_tensor([-1] + loss_cropping + [-1], dtype='int32')),
                                       name='cropping_pred_spread')(spreads_tensor)

    # metrics is computed as part of the model
    if metrics == 'laplace':
        err_tensor = KL.Subtract()([intensities_tensor, image_gt])
        b_tensor = KL.Lambda(lambda x: 1e-5 + 0.02 * tf.exp(x), name='predicted_bs')(spreads_tensor)
        loss_tensor = KL.Lambda(lambda x: K.mean(tf.math.log(2*x[0]) + (K.abs(x[1]) / x[0])),
                                name='laplace_loss')([b_tensor, err_tensor])
    elif metrics == 'l2':
        err_tensor = KL.Subtract()([intensities_tensor, image_gt])
        loss_tensor = KL.Lambda(lambda x: K.mean(K.square(x)), name='L2_loss')(err_tensor)
    elif metrics == 'l1':
        err_tensor = KL.Subtract()([intensities_tensor, image_gt])
        loss_tensor = KL.Lambda(lambda x: K.mean(K.abs(x)), name='L1_loss')(err_tensor)
    elif metrics == 'ssim':

        # TODO: true 3D

        # TODO: multiple output channels
        if image_gt.get_shape()[-1] > 1:
            raise Exception('SSIM metric does not currently support multiple channels')

        ssim_xy = KL.Lambda(
            lambda x: tf.image.ssim(x[0], x[1],
                                    1.0), name='ssim_xy')([intensities_tensor, image_gt])
        ssim_xz = KL.Lambda(
            lambda x: tf.image.ssim(tf.transpose(x[0], perm=[0, 1, 3, 2, 4]), tf.transpose(x[1], perm=[0, 1, 3, 2, 4]),
                                    1.0), name='ssim_xz')([intensities_tensor, image_gt])
        ssim_yz = KL.Lambda(
            lambda x: tf.image.ssim(tf.transpose(x[0], perm=[0, 2, 3, 1, 4]), tf.transpose(x[1], perm=[0, 2, 3, 1, 4]),
                                    1.0), name='ssim_yz')([intensities_tensor, image_gt])

        loss_tensor = KL.Lambda(
            lambda x: -(1 / 3) * tf.reduce_mean(x[0]) - (1 / 3) * tf.reduce_mean(x[1]) - (1 / 3) * tf.reduce_mean(x[2]),
            name='ssim_loss')([ssim_xy, ssim_xz, ssim_yz])

    else:
        raise Exception('metrics should either be "l1" or "l2" or "ssim" oro "laplace", got {}'.format(metrics))

    # create the model and return
    model = Model(inputs=input_model.inputs, outputs=loss_tensor)
    return model


# Add pretrained segmentation CNN to model to regularize synthesis
def add_seg_loss_to_model(input_model,
                          seg_model,
                          generation_labels,
                          segmentation_label_equivalency,
                          rel_weight,
                          loss_cropping,
                          m=None,
                          M=None,
                          fs_header=False):

    # get required layers from input models
    image_loss = input_model.outputs[0]
    predicted_image = input_model.get_layer('predicted_image').output
    segm_target = input_model.get_layer('segmentation_target').output

    # normalise/clip predicted image if needed
    if m is None:
        input_norm = KL.Lambda(lambda x: x + .0, name='input_normalized')(predicted_image)
    else:
        input_norm = KL.Lambda(lambda x: (K.clip(x, m, M) - m) / (M - m), name='input_normalized')(predicted_image)

    # Push predicted image through segmentation CNN
    if fs_header:
        input_normalized_rotated = KL.Lambda(lambda x: K.reverse(K.permute_dimensions(x, [0, 1, 3, 2, 4]), axes=2),
                                             name='input_normalized_rotated')(input_norm)
        predicted_seg_rotated = seg_model(input_normalized_rotated)
        predicted_seg = KL.Lambda(lambda x: K.permute_dimensions(K.reverse(x, axes=2),
                                                                 [0, 1, 3, 2, 4]))(predicted_seg_rotated)
    else:
        predicted_seg = seg_model(input_norm)

    # crop output to evaluate loss function in centre patch
    if loss_cropping is not None:
        # format loss_cropping
        target_shape = predicted_image.get_shape().as_list()[1:-1]
        n_dims, _ = utils.get_dims(target_shape)
        loss_cropping = utils.reformat_to_list(loss_cropping, length=n_dims)

        # perform cropping
        begin_idx = [int((target_shape[i] - loss_cropping[i]) / 2) for i in range(n_dims)]
        segm_target = KL.Lambda(lambda x: tf.slice(x,
                                                   begin=tf.convert_to_tensor([0] + begin_idx + [0], dtype='int32'),
                                                   size=tf.convert_to_tensor([-1] + loss_cropping + [-1],
                                                                             dtype='int32')))(segm_target)
        predicted_seg = KL.Lambda(lambda x: tf.slice(x,
                                                     begin=tf.convert_to_tensor([0] + begin_idx + [0], dtype='int32'),
                                                     size=tf.convert_to_tensor([-1] + loss_cropping + [-1],
                                                                               dtype='int32')))(predicted_seg)

    # reformat gt to have the same label values as for the segmentations
    segmentation_label_equivalency = utils.load_array_if_path(segmentation_label_equivalency)
    generation_labels = utils.load_array_if_path(generation_labels)
    gt_onehot = list()
    pred_onehot = list()
    for i in range(len(generation_labels)):
        idx = np.where(segmentation_label_equivalency == generation_labels[i])[0]
        if len(idx) > 0:
            tensor = KL.Lambda(lambda x: tf.cast(x[..., -1] == i, dtype='float32'))(segm_target)
            gt_onehot.append(tensor)
            if len(idx) == 1:
                tensor2 = KL.Lambda(lambda x: x[..., idx[0]])(predicted_seg)
            elif len(idx) == 2:
                tensor2 = KL.Lambda(lambda x: x[..., idx[0]] + x[..., idx[1]])(predicted_seg)
            elif len(idx) == 3:
                tensor2 = KL.Lambda(lambda x: x[..., idx[0]] + x[..., idx[1]] + x[..., idx[2]])(predicted_seg)
            else:
                raise Exception("uuummm weird that you're merging so many labels...")
            pred_onehot.append(tensor2)
    gt = KL.Lambda(lambda x: tf.stack(x, -1), name='gt')(gt_onehot) if len(gt_onehot) > 1 else gt_onehot[0]
    pred = KL.Lambda(lambda x: tf.stack(x, -1), name='pred')(pred_onehot) if len(pred_onehot) > 1 else pred_onehot[0]

    # Dice loss: it's crucial to disable the checks, so we can use incomplete segmentations
    dice_loss = layers.DiceLoss(enable_checks=False, name='dice_loss')([gt, pred])

    total_loss = KL.Lambda(lambda x: x[0] + rel_weight * x[1])([image_loss, dice_loss])

    # create the model and return
    model = Model(inputs=input_model.inputs, outputs=total_loss)

    return model


class IdentityLoss(object):
    """Very simple loss, as the computation of the loss as been directly implemented in the model."""
    def __init__(self, keepdims=True):
        self.keepdims = keepdims

    def loss(self, y_true, y_predicted):
        """Because the metrics is already calculated in the model, we simply return y_predicted.
           We still need to put y_true in the inputs, as it's expected by keras."""
        loss = y_predicted

        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss
