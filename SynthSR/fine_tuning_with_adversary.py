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
import os
import numpy as np

# tensorfow/keras imports
import tensorflow as tf
from keras import models
import keras.layers as KL
import keras.backend as K
from keras.layers import Layer
from keras.optimizers import Adam

# project imports
from .brain_generator import BrainGenerator

# third-party imports
from ext.lab2im import utils
from ext.neuron import models as nrn_models


def training(labels_dir,
             images_dir,
             model_dir,
             prior_means,
             prior_stds,
             path_generation_labels,
             prior_distributions='normal',
             path_generation_classes=None,
             FS_sort=True,
             batchsize=1,
             input_channels=True,
             output_channel=None,
             target_res=None,
             output_shape=None,
             flipping=True,
             padding_margin=None,
             scaling_bounds=0.2,
             rotation_bounds=20,
             shearing_bounds=0.03,
             translation_bounds=5,
             nonlin_std=5.,
             nonlin_shape_factor=0.04,
             simulate_registration_error=False,
             data_res=None,
             thickness=None,
             randomise_res=True,
             downsample=True,
             blur_range=1.03,
             build_reliability_maps=False,
             bias_field_std=.4,
             bias_shape_factor=0.04,
             n_levels=5,
             nb_conv_per_level=2,
             conv_size=3,
             unet_feat_count=24,
             feat_multiplier=2,
             dropout=0,
             activation='elu',
             lr=1e-4,
             lr_decay=0,
             epochs=100,
             steps_per_epoch=1000,
             training_ratio=10,
             regression_metric='l1',
             work_with_residual_channel=None,
             loss_cropping=None,
             checkpoint_generator=None,
             gradient_penalty_weight=10):
    """
    This function trains a Unet to do slice imputation (and possibly synthesis) of MRI images with thick slices,
    using synthetic scans and possibly real scans.

    :param labels_dir: path of folder with all input label maps, or to a single label map (if only one training example)
    :param model_dir: path of a directory where the models will be saved during training.
    :param images_dir: directory with real images corresponding to the training label maps. These will be taken as
    regression target. We recommend skull stripping them.

    #---------------------------------------------- Generation parameters ----------------------------------------------
    # label maps parameters
    :param path_generation_labels: list of all possible label values in the input label maps.
    Must be the path to a 1d numpy array, which should be organised as follows: background label first, then non-sided
    labels (e.g. CSF, brainstem, etc.), then all the structures of the same hemisphere (can be left or right), and
    finally all the corresponding contralateral structures (in the same order).
    Example: [background_label, non-sided_1, ..., non-sided_n, left_1, ..., left_m, right_1, ..., right_m]
    :param FS_sort: whether us FS_sort when creating list of labels with utils.get_list_labels. Default is True.

    # output-related parameters
    :param batchsize: (optional) number of images to generate per mini-batch. Default is 1.
    :param input_channels: (optional) list of booleans indicating if each *synthetic* channel is going to be used as an
    input for the downstream network. This also enables to know how many channels are going to be synthesised. Default
    is True, which means generating 1 channel, and use it as input (either for plain SR with a synthetic target, or for
    synthesis with a real target).
    :param output_channel: (optional) a list with the indices of the output channels  (i.e. the synthetic regression
    targets), if no real images were provided as regression target. Set to None if using real images as targets. Default
    is the first channel (index 0).
    :param target_res: (optional) target resolution of the generated images and corresponding label maps.
    If None, the outputs will have the same resolution as the input label maps.
    Can be a number (isotropic resolution), or the path to a 1d numpy array.
    :param output_shape: (optional) desired shape of the output image, obtained by randomly cropping the generated image
    Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.
    Default is None, where no cropping is performed.

    # GMM-sampling parameters
    :param path_generation_classes: (optional) Indices regrouping generation labels into classes of same intensity
    distribution. Regouped labels will thus share the same Gaussian when samling a new image. Should be the path to a 1d
    numpy array with the same length as generation_labels. and contain values between 0 and K-1, where K is the total
    number of classes. Default is all labels have different classes.
    :param prior_distributions: (optional) type of distribution from which we sample the GMM parameters.
    Can either be 'uniform', or 'normal'. Default is 'normal'.
    :param prior_means: (optional) hyperparameters controlling the prior distributions of the GMM means. Because
    these prior distributions are uniform or normal, they require by 2 hyperparameters. Can be a path to:
    1) an array of shape (2, K), where K is the number of classes (K=len(generation_labels) if generation_classes is
    not given). The mean of the Gaussian distribution associated to class k in [0, ...K-1] is sampled at each mini-batch
    from U(prior_means[0,k], prior_means[1,k]) if prior_distributions is uniform, and from
    N(prior_means[0,k], prior_means[1,k]) if prior_distributions is normal.
    2) an array of shape (2*n_mod, K), where each block of two rows is associated to hyperparameters derived
    from different modalities. In this case, if use_specific_stats_for_channel is False, we first randomly select a
    modality from the n_mod possibilities, and we sample the GMM means like in 2).
    If use_specific_stats_for_channel is True, each block of two rows correspond to a different channel
    (n_mod=n_channels), thus we select the corresponding block to each channel rather than randomly drawing it.
    Default is None, which corresponds all GMM means sampled from uniform distribution U(25, 225).
    :param prior_stds: (optional) same as prior_means but for the standard deviations of the GMM.
    Default is None, which corresponds to U(5, 25).

    # spatial deformation parameters
    :param flipping: (optional) whether to introduce right/left random flipping. Default is True.
    :param  padding_margin: useful when cropping the loss but you are not using very large patches. Set to None for
    determining it automatically from loss_cropping (not recommended if you use big volume sizes)
    :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
    sampled from a uniform distribution of predefined bounds. Can either be:
    1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
    (1-scaling_bounds, 1+scaling_bounds) for each dimension.
    2) the path to a numpy array of shape (2, n_dims), in which case the scaling factor in dimension i is sampled from
    the uniform distribution of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
    3) False, in which case scaling is completely turned off.
    Default is scaling_bounds = 0.15 (case 1)
    :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for case 1 the
    bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
    Default is rotation_bounds = 15.
    :param shearing_bounds: (optional) same as scaling bounds. Default is shearing_bounds = 0.012.
    :param translation_bounds: (optional) same as scaling bounds. Default is translation_bounds = False, but we
    encourage using it when cropping is deactivated (i.e. when output_shape=None).
    :param nonlin_std: (optional) Standard deviation of the normal distribution from which we sample the first
    tensor for synthesising the deformation field. Set to 0 to completely turn the elastic deformation off.
    :param nonlin_shape_factor: (optional) Ratio between the size of the input label maps and the size of the sampled
    tensor for synthesising the elastic deformation field.
    :param simulate_registration_error: (optional) whether to simulate registration errors between *synthetic* channels.
    Can be a single value (same for all channels) or a list with one value per *synthetic* channel. For the latter,
    the first value will automatically be reset to True since the first channel is used as reference. Default is True.

    # blurring/resampling parameters
    :param randomise_res: (optional) whether to mimic images that would have been 1) acquired at low resolution, and
    2) resampled to high resolution. The low resolution is uniformly sampled at each minibatch from [1mm, 9mm].
    In that process, the images generated by sampling the GMM are: 1) blurred at LR, 2) downsampled at LR, and
    3) resampled at target_resolution.
    :param data_res: (optional) specific acquisition resolution to mimic, as opposed to random resolution sampled when
    randomis_res is True. This triggers a blurring which mimics the acquisition resolution, but downsampling is optional
    (see param downsample). Default for data_res is None, where images are slighlty blurred. If the generated images are
    uni-modal, data_res can be a number (isotropic acquisition resolution), a sequence, a 1d numpy array, or the path
    to a 1d numy array. In the multi-modal case, it should be given as a umpy array (or a path) of size (n_mod, n_dims),
    where each row is the acquisition resolution of the corresponding channel.
    :param thickness: (optional) if data_res is provided, we can further specify the slice thickness of the low
    resolution images to mimic. Must be provided in the same format as data_res. Default thickness = data_res.
    :param downsample: (optional) whether to actually downsample the volume images to data_res after blurring.
    Default is False, except when thickness is provided, and thickness < data_res.
    :param blur_range: (optional) Randomise the standard deviation of the blurring kernels, (whether data_res is given
    or not). At each mini_batch, the standard deviation of the blurring kernels are multiplied by a coefficient sampled
    from a uniform distribution with bounds [1/blur_range, blur_range]. If None, no randomisation. Default is 1.15.
    :param build_reliability_maps: set to True if you want to build soft masks indicating which voxels are
    "measured" and which are interpolated

    # bias field parameters
    :param bias_field_std: (optional) If strictly positive, this triggers the corruption of synthesised images with a
    bias field. This will only affect the input channels (i.e. not the synthetic regression target). The bias field is
    obtained by sampling a first small tensor from a normal distribution, resizing it to full size, and rescaling it to
    positive values by taking the voxel-wise exponential. bias_field_std designates the std dev of the normal
    distribution from which we sample the first tensor. Set to 0 to completely deactivate biad field corruption.
    :param bias_shape_factor: (optional) If bias_field_std is not False, this designates the ratio between the size of
    the input label maps and the size of the first sampled tensor for synthesising the bias field.

    # ------------------------------------------ UNet architecture parameters ------------------------------------------
    :param n_levels: (optional) number of level for the Unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutional layers per level. Default is 2.
    :param conv_size: (optional) size of the convolution kernels. Default is 2.
    :param unet_feat_count: (optional) number of feature for the first layr of the Unet. Default is 24.
    :param feat_multiplier: (optional) multiply the number of feature by this nummber at each new level. Default is 2.
    :param dropout: (optional) probability of dropout for the Unet. Deafult is 0, where no dropout is applied.
    :param activation: (optional) activation function. Can be 'elu', 'relu'.

    # ----------------------------------------------- Training parameters ----------------------------------------------
    :param lr: (optional) learning rate for the training. Default is 1e-4
    :param lr_decay: (optional) learing rate decay. Default is 0, where no decay is applied.
    :param epochs: (optional) number of epochs.
    :param steps_per_epoch: (optional) number of steps per epoch. Default is 1000. Since no online validation is
    possible, this is equivalent to the frequency at which the models are saved.
    :param training_ratio: (optional) number of discriminator iterations to take at each training step (whereas the
    generator is only iterated once per step). This doesn't apply to the first step of the first epoch, where the
    discriminator is trained for 1,000 iterations. Default is 10.
    :param regression_metric: (optional) loss used in training. Can be 'l1' (default), 'l2', 'ssim', or 'laplace'
    :param work_with_residual_channel: (optional) if you have a channel that is similar to the output (e.g., in
    imputation), it is convenient to predict the residual, rather than the image from scratch. This parameter is a list
    of indices of the synthetic channels you want to add the residual to (must have the same length as output_channels,
    or have length equal to 1 if real images are used)
    :param loss_cropping: (option)  to crop the posteriors when evaluating the loss function (specify the output size
    Can be an int, or the path to a 1d numpy array.
    """

    n_channels = len(utils.reformat_to_list(input_channels))

    # convert output_channel and work_with_residual_channel to lists
    if output_channel is not None:
        output_channel = list(utils.reformat_to_list(output_channel))
        n_output_channels = len(output_channel)
    else:
        n_output_channels = 1

    # various checks
    if (images_dir is None) & (output_channel is None):
        raise Exception('please provide a value for output_channel or image_dir')
    elif (images_dir is not None) & (output_channel is not None):
        raise Exception('please provide a value either for output_channel or image_dir, but not both at the same time')
    if output_channel is not None:
        if any(x >= n_channels for x in output_channel):
            raise Exception('indices in output_channel cannot be greater than the total number of channels')

    # check work_with_residual_channel
    if work_with_residual_channel is not None:
        work_with_residual_channel = utils.reformat_to_list(work_with_residual_channel)
        if output_channel is not None:
            if len(work_with_residual_channel) != len(output_channel):
                raise Exception('The number or residual channels and output channels must be the same')

        if any(x >= n_channels for x in work_with_residual_channel):
            raise Exception('indices in work_with_residual_channel cannot be greater than the total number of channels')

    # get label lists
    generation_labels, n_neutral_labels = utils.get_list_labels(label_list=path_generation_labels,
                                                                labels_dir=labels_dir,
                                                                FS_sort=FS_sort)

    # prepare model folder
    utils.mkdir(model_dir)
    log_dir = os.path.join(model_dir, 'logs')
    utils.mkdir(log_dir)

    # compute padding_margin if needed
    if loss_cropping == 0:
        padding_margin = None
    elif padding_margin is None:
        padding_margin = utils.get_padding_margin(output_shape, loss_cropping)

    # instantiate BrainGenerator object
    brain_generator = BrainGenerator(labels_dir=labels_dir,
                                     images_dir=images_dir,
                                     generation_labels=generation_labels,
                                     n_neutral_labels=n_neutral_labels,
                                     padding_margin=padding_margin,
                                     batchsize=batchsize,
                                     input_channels=input_channels,
                                     output_channel=output_channel,
                                     target_res=target_res,
                                     output_shape=output_shape,
                                     output_div_by_n=2 ** n_levels,
                                     generation_classes=path_generation_classes,
                                     prior_means=prior_means,
                                     prior_stds=prior_stds,
                                     prior_distributions=prior_distributions,
                                     flipping=flipping,
                                     scaling_bounds=scaling_bounds,
                                     rotation_bounds=rotation_bounds,
                                     shearing_bounds=shearing_bounds,
                                     translation_bounds=translation_bounds,
                                     nonlin_std=nonlin_std,
                                     nonlin_shape_factor=nonlin_shape_factor,
                                     simulate_registration_error=simulate_registration_error,
                                     randomise_res=randomise_res,
                                     data_res=data_res,
                                     thickness=thickness,
                                     downsample=downsample,
                                     blur_range=blur_range,
                                     build_reliability_maps=build_reliability_maps,
                                     bias_field_std=bias_field_std,
                                     bias_shape_factor=bias_shape_factor)

    # input generator
    input_generator = utils.build_training_generator(brain_generator.model_inputs_generator, batchsize)

    # transformation model
    labels_to_image_model = brain_generator.labels_to_image_model
    unet_input_shape = brain_generator.model_output_shape

    # generator model
    generator = nrn_models.unet(nb_features=unet_feat_count,
                                 input_shape=unet_input_shape,
                                 nb_levels=n_levels,
                                 conv_size=conv_size,
                                 nb_labels=2*n_output_channels if regression_metric == 'laplace' else n_output_channels,
                                 feat_mult=feat_multiplier,
                                 nb_conv_per_level=nb_conv_per_level,
                                 conv_dropout=dropout,
                                 final_pred_activation='linear',
                                 batch_norm=-1,
                                 activation=activation,
                                 input_model=labels_to_image_model)
    if checkpoint_generator is not None:
        print('loading', checkpoint_generator)
        generator.load_weights(checkpoint_generator, by_name=True)

    # discriminator model
    discriminator = make_discriminator(unet_input_shape)

    # add frozen discriminator to generator for training
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    generator_out = generator.output
    generator_discriminator_out = discriminator(generator_out)

    # add loss computation to model, because the real image is modified (= cropped) inside the generation model,
    # loss = 0.999 l1_loss + 0.001 wasserstein_loss
    target = generator.get_layer('regression_target').output
    gen_loss = build_generator_loss(target, generator_out, generator_discriminator_out, loss_cropping)

    # build and compile generator model
    generator_model = models.Model(inputs=generator.inputs, outputs=gen_loss)
    generator_model.compile(optimizer=Adam(learning_rate=lr, decay=lr_decay), loss=dummy_loss)

    # freeze generator when training discriminator
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    # define discriminator inputs
    target = generator.get_layer('regression_target').output
    generated_samples_for_discriminator = generator.output
    averaged_samples = RandomWeightedAverage()([target, generated_samples_for_discriminator])

    # define discriminator outputs
    discriminator_real = discriminator(target)
    discriminator_fake = discriminator(generated_samples_for_discriminator)
    discriminator_av = discriminator(averaged_samples)

    # add discriminator loss to model
    discr_loss = build_discriminator_loss(discriminator_real, discriminator_fake, discriminator_av,
                                                  averaged_samples, gradient_penalty_weight, brain_generator.n_dims)

    # create discriminator model
    discriminator_model = models.Model(inputs=generator.inputs, outputs=discr_loss)
    discriminator_model.compile(optimizer=Adam(learning_rate=lr, decay=lr_decay), loss=dummy_loss)

    # training loop
    le = len(str(epochs))
    ls = len(str(steps_per_epoch))
    discriminator_logs = np.array([])
    generator_logs = np.array([])
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))

        avg_discr_loss = 0
        avg_gen_loss = 0
        for step in range(int(steps_per_epoch)):

            # take several training steps for discriminator
            tmp_training_ratio = 1000 if (epoch == 0) & (step == 0) else training_ratio
            for j in range(tmp_training_ratio):
                training_inputs, dummy_y = next(input_generator)
                discr_loss = discriminator_model.train_on_batch(training_inputs, dummy_y)
                avg_discr_loss += (discr_loss / (steps_per_epoch * tmp_training_ratio))
                print('{0:0{1}d}/{2}   discriminator loss:   {3}'.format(step+1, ls, steps_per_epoch, discr_loss))

            # take a step in generator
            training_inputs, dummy_y = next(input_generator)
            gen_loss = generator_model.train_on_batch(training_inputs, dummy_y)
            avg_gen_loss += (gen_loss / steps_per_epoch)
            print('{0:0{1}d}/{2}   generator loss:       {3}'.format(step + 1, ls, steps_per_epoch, gen_loss))

        # print and save epoch metrics
        print('Epoch {0:0{1}d}/{2}   average discriminator loss:   {3}'.format(epoch + 1, le, epochs, avg_discr_loss))
        print('Epoch {0:0{1}d}/{2}   average generator loss:       {3}'.format(epoch + 1, le, epochs, avg_gen_loss))
        discriminator_logs = np.append(discriminator_logs, avg_discr_loss)
        generator_logs = np.append(generator_logs, avg_gen_loss)
        np.save(os.path.join(log_dir, 'discriminator_loss.npy'), discriminator_logs)
        np.save(os.path.join(log_dir, 'generator_loss.npy'), generator_logs)

        # save model
        print('Epoch {0:0{1}d}/{2}   saving models\n'.format(epoch + 1, le, epochs))
        generator_model.save(os.path.join(model_dir, 'generator_{0:0{1}d}.h5'.format(epoch + 1, le)))
        discriminator_model.save(os.path.join(model_dir, 'discriminator_{0:0{1}d}.h5'.format(epoch + 1, le)))


def make_discriminator(input_shape, n_filters=32, n_levels=4):

    input_tensor = KL.Input(shape=input_shape, name='input_discriminator')
    last_tensor = input_tensor

    for level in range(n_levels):
        last_tensor = discriminator_block(last_tensor, n_filters * (2 ** level), strides=1)
        last_tensor = discriminator_block(last_tensor, n_filters * (2 ** level), strides=2)

    last_tensor = KL.Flatten(data_format='channels_last')(last_tensor)
    last_tensor = KL.Dense(n_filters * (2 ** n_levels))(last_tensor)
    last_tensor = KL.LeakyReLU(alpha=0.2)(last_tensor)

    # output without activation
    last_tensor = KL.Dense(1, activation=None)(last_tensor)

    return models.Model(input_tensor, last_tensor)


def discriminator_block(layer_input, filters, strides):
    d = KL.Conv3D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
    d = KL.LeakyReLU(alpha=0.2)(d)
    return d


def build_generator_loss(target, generator_output, generator_discriminator_output, loss_cropping=16):

    # crop tensors to compute loss only in the middle
    if loss_cropping is not None:

        # format loss_cropping
        target_shape = target.get_shape().as_list()[1:-1]
        n_dims, _ = utils.get_dims(target_shape)
        loss_cropping = utils.reformat_to_list(loss_cropping, length=n_dims)

        # perform cropping
        idx = [int((target_shape[i] - loss_cropping[i]) / 2) for i in range(n_dims)]
        target = KL.Lambda(lambda x: tf.slice(x, begin=tf.convert_to_tensor([0] + idx + [0], dtype='int32'),
                           size=tf.convert_to_tensor([-1] + loss_cropping + [-1], dtype='int32')),
                           name='cropping_gt')(target)
        generator_output = KL.Lambda(lambda x: tf.slice(x, begin=tf.convert_to_tensor([0] + idx + [0], dtype='int32'),
                                     size=tf.convert_to_tensor([-1] + loss_cropping + [-1], dtype='int32')),
                                     name='cropping_pred')(generator_output)

    # create loss for generator
    l1_loss = KL.Lambda(lambda x: K.mean(K.abs(x[0] - x[1])), name='L1_loss')([target, generator_output])
    w_loss = KL.Lambda(lambda x: K.mean(-x), name='w_loss')(generator_discriminator_output)
    generator_loss = KL.Lambda(lambda x: 0.999 * x[0] + 0.001 * x[1], name='gen_loss')([l1_loss, w_loss])

    return generator_loss


def build_discriminator_loss(discriminator_real, discriminator_fake, discriminator_av, averaged_samples,
                             gradient_penalty_w=10, n_dims=3):

    # compute gradient penalty
    gradients = Gradients()([discriminator_av, averaged_samples])
    gradients_l2_norm = KL.Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=np.arange(1, n_dims + 1))))(gradients)
    gradient_penalty = KL.Lambda(lambda x: gradient_penalty_w * K.square(1 - x))(gradients_l2_norm)

    # create loss for discriminator
    w_loss_real = KL.Lambda(lambda x: K.mean(-x), name='w_loss_real')(discriminator_real)
    w_loss_fake = KL.Lambda(lambda x: K.mean(x), name='w_loss_fake')(discriminator_fake)
    gradient_penalty_loss = KL.Lambda(lambda x: K.mean(x), name='gradient_penalty_loss')(gradient_penalty)
    discriminator_loss = KL.Lambda(lambda x: x[0] + x[1] + x[2], name='discriminator_loss')([w_loss_real,
                                                                                             w_loss_fake,
                                                                                             gradient_penalty_loss])

    return discriminator_loss


def dummy_loss(y_true, y_predicted):
    """Because the metrics is already calculated in the model, we simply return y_predicted.
       We still need to put y_true in the inputs, as it's expected by keras."""
    return y_predicted


class RandomWeightedAverage(Layer):

    def __init__(self, **kwargs):
        self.shape = None
        self.n_dims = None
        super(RandomWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = tuple(input_shape[0])
        self.n_dims = len(self.shape) - 2
        self.built = True
        super(RandomWeightedAverage, self).build(input_shape)

    def call(self, inputs, **kwargs):
        batchsize = tf.split(tf.shape(inputs[0]), [1, -1])[0]
        sample_shape = tf.concat([batchsize, tf.ones([self.n_dims + 1], dtype='int32')], 0)
        weights = tf.random.uniform(sample_shape, maxval=1.)
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

    def compute_output_shape(self, input_shape):
        return self.shape


class Gradients(Layer):

    def __init__(self, **kwargs):
        self.shape = None
        super(Gradients, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = tuple(input_shape[1])
        self.built = True
        super(Gradients, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return K.gradients(inputs[0], inputs[1])[0]

    def compute_output_shape(self, input_shape):
        return self.shape
