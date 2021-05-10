# python imports
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model
import numpy as np

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_tensors as l2i_et
from ext.lab2im import layers


def metrics_model(input_model, loss_cropping=16, metrics='l1', work_with_residual_channel=None):

    # first layer: input
    last_tensor = input_model.outputs[0]

    # add residual if needed
    if work_with_residual_channel is None:
        last_tensor = KL.Lambda(lambda x: x, name='predicted_image')(last_tensor)
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
        last_tensor = KL.Add(name='predicted_image')([slices, last_tensor])

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
        last_tensor = KL.Lambda(lambda x: tf.slice(x, begin=tf.convert_to_tensor([0] + begin_idx + [0], dtype='int32'),
                                size=tf.convert_to_tensor([-1] + loss_cropping + [-1], dtype='int32')),
                                name='cropping_pred')(last_tensor)

    # metrics is computed as part of the model
    if metrics == 'l2':
        last_tensor = KL.Subtract()([last_tensor, image_gt])
        last_tensor = KL.Lambda(lambda x: K.mean(K.square(x)), name='L2_loss')(last_tensor)
    elif metrics == 'l1':
        last_tensor = KL.Subtract()([last_tensor, image_gt])
        last_tensor = KL.Lambda(lambda x: K.mean(K.abs(x)), name='L1_loss')(last_tensor)
    elif metrics == 'ssim':

        # TODO: true 3D

        # TODO: multiple output channels
        if image_gt.get_shape()[-1] > 1:
            raise Exception('SSIM metric does not currently support multiple channels')

        ssim_xy = KL.Lambda(
            lambda x: tf.image.ssim(x[0], x[1],
                                    1.0), name='ssim_xy')([last_tensor, image_gt])
        ssim_xz = KL.Lambda(
            lambda x: tf.image.ssim(tf.transpose(x[0], perm=[0, 1, 3, 2, 4]), tf.transpose(x[1], perm=[0, 1, 3, 2, 4]),
                                    1.0), name='ssim_xz')([last_tensor, image_gt])
        ssim_yz = KL.Lambda(
            lambda x: tf.image.ssim(tf.transpose(x[0], perm=[0, 2, 3, 1, 4]), tf.transpose(x[1], perm=[0, 2, 3, 1, 4]),
                                    1.0), name='ssim_yz')([last_tensor, image_gt])

        last_tensor = KL.Lambda(
            lambda x: -(1 / 3) * tf.reduce_mean(x[0]) - (1 / 3) * tf.reduce_mean(x[1]) - (1 / 3) * tf.reduce_mean(x[2]),
            name='ssim_loss')([ssim_xy, ssim_xz, ssim_yz])

    else:
        raise Exception('metrics should either be "l1" or "l2" or "ssim", got {}'.format(metrics))

    # create the model and return
    model = Model(inputs=input_model.inputs, outputs=last_tensor)
    return model


# Add pretrained segmentation CNN to model to regularize synthesis
def add_seg_loss_to_model(input_model, seg_model, generation_labels, segmentation_label_equivalency, rel_weight, loss_cropping, mini=-1e10, maxi=1e10):

    # get required layers from input models
    image_loss = input_model.outputs[0]
    predicted_image = input_model.get_layer('predicted_image').output
    segmentation_target = input_model.get_layer('segmentation_target').output

    # Push predicted image through segmentation CNN (requires clipping / normalization)
    input_normalized = KL.Lambda(lambda x: (K.clip(x, mini, maxi) - mini) / (maxi - mini), name='input_normalized')(predicted_image)

    predicted_seg = seg_model(input_normalized)

    # crop output to evaluate loss function in centre patch
    if loss_cropping is not None:
        # format loss_cropping
        target_shape = predicted_image.get_shape().as_list()[1:-1]
        n_dims, _ = utils.get_dims(target_shape)
        loss_cropping = utils.reformat_to_list(loss_cropping, length=n_dims)

        # perform cropping
        begin_idx = [int((target_shape[i] - loss_cropping[i]) / 2) for i in range(n_dims)]
        segmentation_target = KL.Lambda(lambda x: tf.slice(x, begin=tf.convert_to_tensor([0] + begin_idx + [0], dtype='int32'),
                                                size=tf.convert_to_tensor([-1] + loss_cropping + [-1], dtype='int32')))(segmentation_target)
        predicted_seg = KL.Lambda(lambda x: tf.slice(x, begin=tf.convert_to_tensor([0] + begin_idx + [0], dtype='int32'),
                                                size=tf.convert_to_tensor([-1] + loss_cropping + [-1], dtype='int32')))(predicted_seg)



    if type(segmentation_label_equivalency) is str:
        segmentation_label_equivalency = np.load(segmentation_label_equivalency)
    if type(generation_labels) is str:
        generation_labels = np.load(generation_labels)

    gt_onehot = list()
    pred_onehot = list()
    for i in range(len(generation_labels)):
        idx = np.where(segmentation_label_equivalency==generation_labels[i])[0]
        if len(idx)>0:
            tensor = KL.Lambda(lambda x: tf.cast(x[...,-1]==i,dtype='float32'))(segmentation_target)
            gt_onehot.append(tensor)

            if len(idx)==1:
                tensor2 = KL.Lambda(lambda x: x[..., idx[0]])(predicted_seg)
            elif len(idx)==2:
                tensor2 = KL.Lambda(lambda x: x[..., idx[0]] + x[..., idx[1]])(predicted_seg)
            elif len(idx)==3:
                tensor2 = KL.Lambda(lambda x: x[..., idx[0]] + x[..., idx[1]] + x[..., idx[2]])(predicted_seg)
            else:
                raise Exception("uuummm weird that you're merging so many labels...")
            pred_onehot.append(tensor2)

    gt_tensor = KL.Lambda(lambda x: tf.stack(x, -1), name='gt_tensor')(gt_onehot) if len(gt_onehot) > 1 else gt_onehot[0]
    pred_tensor = KL.Lambda(lambda x: tf.stack(x, -1), name='pred_tensor')(pred_onehot) if len(pred_onehot) > 1 else pred_onehot[0]

    # Dice loss: it's crucial to disable the checks, so we can use incomplete segmentations
    dice_loss = layers.DiceLoss(enable_checks=False)([gt_tensor, pred_tensor], name='dice_loss')

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
