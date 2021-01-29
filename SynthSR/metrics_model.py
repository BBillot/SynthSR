# python imports
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model

# third-party imports
from ext.lab2im import utils


def metrics_model(input_shape,
                  input_model=None,
                  loss_cropping=16,
                  metrics='l1',
                  name=None,
                  work_with_residual_channel=None):

    # naming the model
    model_name = name

    # first layer: input
    name = '%s_input' % model_name
    if input_model is None:
        input_tensor = KL.Input(shape=input_shape, name=name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.inputs
        last_tensor = input_model.outputs
        if isinstance(last_tensor, list):
            last_tensor = last_tensor[0]
        last_tensor = KL.Reshape(input_shape, name='predicted_output')(last_tensor)

    # add residual if needed
    if work_with_residual_channel is not None:
        slice_list = list()
        for c in work_with_residual_channel:
            tensor = input_model.get_layer('image_out').output
            tmp_slice = KL.Lambda(lambda x: tf.expand_dims(x[..., c], axis=-1))(tensor)
            slice_list.append(tmp_slice)
        if len(slice_list) > 1:
            slices = KL.Lambda(lambda x: tf.concat(x, axis=-1))(slice_list)
        else:
            slices = slice_list[0]
        last_tensor = KL.Add()([slices, last_tensor])

    # get crisp, ground truth image
    image_gt = input_model.get_layer('regression_target').output
    image_gt = KL.Lambda(lambda x: x, name='target')(image_gt)

    # crop output to evaluate loss function in centre patch
    if loss_cropping is not None:
        # format loss_cropping
        target_shape = image_gt.get_shape().as_list()[1:-1]
        n_dims, _ = utils.get_dims(target_shape)
        if isinstance(loss_cropping, (int, float)):
            loss_cropping = [loss_cropping] * n_dims
        if isinstance(loss_cropping, (list, tuple)):
            if len(loss_cropping) == 1:
                loss_cropping = loss_cropping * n_dims
            elif len(loss_cropping) != n_dims:
                raise TypeError('loss_cropping should be float, list of size 1 or {0}, or None. '
                                'Had {1}'.format(n_dims, loss_cropping))
        # perform cropping
        begin_idx = [int((target_shape[i] - loss_cropping[i]) / 2) for i in range(n_dims)]
        image_gt = KL.Lambda(
            lambda x: tf.slice(x, begin=tf.convert_to_tensor([0] + begin_idx + [0], dtype='int32'),
                               size=tf.convert_to_tensor([-1] + loss_cropping + [-1], dtype='int32')),
            name='cropping_gt')(image_gt)
        last_tensor = KL.Lambda(
            lambda x: tf.slice(x, begin=tf.convert_to_tensor([0] + begin_idx + [0], dtype='int32'),
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
    model = Model(inputs=input_tensor, outputs=last_tensor, name=model_name)
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
