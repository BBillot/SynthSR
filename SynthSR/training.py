# python imports
import os
import keras.callbacks as KC
from keras.models import Model
from keras.optimizers import Adam

# project imports
from .metrics_model import metrics_model, IdentityLoss
from .brain_generator import BrainGenerator

# third-party imports
from ext.lab2im import utils
from ext.neuron import models as nrn_models


def training(labels_dir,
             model_dir,
             prior_means,
             prior_stds,
             path_generation_labels,
             prior_distributions='normal',
             images_dir=None,
             path_generation_classes=None,
             FS_sort=True,
             batchsize=1,
             input_channels=True,
             output_channel=0,
             target_res=None,
             output_shape=None,
             flipping=True,
             padding_margin=None,
             scaling_bounds=0.15,
             rotation_bounds=15,
             shearing_bounds=0.02,
             translation_bounds=5,
             nonlin_std=4.,
             nonlin_shape_factor=0.03125,
             simulate_registration_error=True,
             data_res=None,
             thickness=None,
             downsample=True,
             blur_range=1.15,
             build_reliability_maps=True,
             bias_field_std=.3,
             bias_shape_factor=0.03125,
             n_levels=5,
             nb_conv_per_level=2,
             conv_size=3,
             unet_feat_count=24,
             feat_multiplier=2,
             dropout=0,
             activation='elu',
             learning_rate=1e-4,
             lr_decay=0,
             epochs=100,
             steps_per_epoch=1000,
             regression_metric='l1',
             work_with_residual_channel=None,
             loss_cropping=None,
             load_model_file=None,
             initial_epoch=0):
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
    Can be a single value (same for all channels) or a list with one value per *synthetic* channel. In the latter case,
    the first values will automatically be reset to True as the first channel is used as reference. Default is True.

    # blurring/resampling parameters
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
    :param learning_rate: (optional) learning rate for the training. Default is 1e-4
    :param lr_decay: (optional) learing rate decay. Default is 0, where no decay is applied.
    :param epochs: (optional) number of epochs.
    :param steps_per_epoch: (optional) number of steps per epoch. Default is 1000. Since no online validation is
    possible, this is equivalent to the frequency at which the models are saved.
    :param regression_metric: (optional) loss used in training. Can be 'l1' (default), 'l2', or 'ssim'
    :param work_with_residual_channel: (optional) if you have a channel that is similar to the output (e.g., in
    imputation), it is convenient to predict the residual, rather than the image from scratch. This parameter is a list
    of indices of the synthetic channels you want to add the residual to (must have the same length as output_channels,
    or have length equal to 1 if real images are used)
    :param loss_cropping: (option)  to crop the posteriors when evaluating the loss function (specify the output size
    Can be an int, or the path to a 1d numpy array.
    :param load_model_file: (optional) path of an already saved model to load before starting the training.
    :param initial_epoch: (optional) initial epoch for training. Useful for resuming training.
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

        if build_reliability_maps:  # consider indices of reliability maps
            work_with_residual_channel = 2 * work_with_residual_channel

    # get label lists
    generation_labels, n_neutral_labels = utils.get_list_labels(label_list=path_generation_labels,
                                                                labels_dir=labels_dir,
                                                                FS_sort=FS_sort)

    # prepare model folder
    utils.mkdir(model_dir)

    # prepare log folder
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
                                     data_res=data_res,
                                     thickness=thickness,
                                     downsample=downsample,
                                     blur_range=blur_range,
                                     build_reliability_maps=build_reliability_maps,
                                     bias_field_std=bias_field_std,
                                     bias_shape_factor=bias_shape_factor)

    # transformation model
    labels_to_image_model = brain_generator.labels_to_image_model
    unet_input_shape = brain_generator.model_output_shape

    # prepare the Unet model
    unet_model = nrn_models.unet(nb_features=unet_feat_count,
                                 input_shape=unet_input_shape,
                                 nb_levels=n_levels,
                                 conv_size=conv_size,
                                 nb_labels=n_output_channels,
                                 feat_mult=feat_multiplier,
                                 nb_conv_per_level=nb_conv_per_level,
                                 conv_dropout=dropout,
                                 final_pred_activation='linear',
                                 batch_norm=-1,
                                 activation=activation,
                                 input_model=labels_to_image_model)

    # input generator
    train_example_gen = brain_generator.model_inputs_generator
    training_generator = utils.build_training_generator(train_example_gen, batchsize)

    # model
    model = Model(unet_model.inputs, [unet_model.get_layer('unet_prediction').output])
    model = metrics_model(input_model=model,
                          loss_cropping=loss_cropping,
                          metrics=regression_metric,
                          work_with_residual_channel=work_with_residual_channel)
    if load_model_file is not None:
        print('loading', load_model_file)
        model.load_weights(load_model_file, by_name=True)
    train_model(model, training_generator, learning_rate, lr_decay, epochs, steps_per_epoch, model_dir, log_dir,
                initial_epoch)


def train_model(model,
                generator,
                learning_rate,
                lr_decay,
                n_epochs,
                n_steps,
                model_dir,
                log_dir,
                initial_epoch=0):

    # model callbacks
    save_file_name = os.path.join(model_dir, '{epoch:03d}.h5')
    callbacks = [KC.ModelCheckpoint(save_file_name, save_weights_only=True, verbose=1),
                 KC.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)]

    # compile
    model.compile(optimizer=Adam(lr=learning_rate, decay=lr_decay),
                  loss=IdentityLoss().loss,
                  loss_weights=[1.0])

    # fit
    model.fit_generator(generator,
                        epochs=n_epochs,
                        steps_per_epoch=n_steps,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch)
