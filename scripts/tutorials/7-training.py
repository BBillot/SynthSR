"""This script shows how to call the training function. It re-uses the last example (6-SR_synthesis_synthetic), where
we use T1 and T2 synthetic as input channels (at HR but simulating data acquired at LR) and synthetic HR T1 scans as
regression target.


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


import numpy as np
from SynthSR.training import training

# we have to specify a model dir, where the models will be saved after each epoch
model_dir = '../../data/generated_images/7-training'  # folder where they will be saved

# we specify the Unet architecture
n_levels = 5  # number of levels in the UNet
nb_conv_per_level = 2  # number of convolution per level
conv_size = 3  # size of the convolution kernels
unet_feat_count = 24  # number of feature maps after the very first layer of the network
# here we double the number of feature maps after each max-pooling operation. Incidentally, the number of features will
# be halved after each upsampling step. Set to 1 to keep the number of feature maps constant throughout the network.
feat_multiplier = 2
dropout = 0  # We recommend not using dropout
activation = 'elu'  # activation function

# we now set the learning parameters
learning_rate = 1e-4  # learning rate to apply
lr_decay = 0   # here we do not use a decay. I fyou do, remember that it will be applied at each step !
# An epoch is defined by a given number of steps (rather than the fact to have gone through all the training examples).
# This choice is motivated by the fact that we typically have a small amount of data in medical imaging analysis.
# At each step, we randomly select a training label map, generate the training data, run the input channels through the
# network, compute the regression metric between the prediction and the regresion target, and finally backpropagate.
# We set here the number of epochs and steps per epoch to low values, as this is just an example, but it would typically
# be 200 epochs with 1,000 steps each.
epochs = 3  # number of epochs
steps_per_epoch = 5  # number of steps per epoch
regression_metric = 'l1'  # metric used to compute the loss function
# In this example, the regression target and one of the input channels have the same contrast (T1 scan). Therefore, it
# is easier to predict the residuals between the two. To do that we need to indicate the index of the input channel to
# add the residuals to
work_with_residual_channel = 1

# We now set the generation parameters, same as before

# data paths
labels_folder = '../../data/labels'
images_folder = None

# general parameters
input_channels = [False, True, True]  # specify which channel will be used as input channel for the network
output_channel = 0   # index corresponding to the regression target
target_res = None
output_shape = 128

# GMM-sampling parameters
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
generation_classes = '../../data/labels_classes_priors/generation_classes.npy'
prior_means_t1_hr = np.load('../../data/labels_classes_priors/prior_means_t1_hr.npy')
prior_means_t1_lr = np.load('../../data/labels_classes_priors/prior_means_t1_lr.npy')
prior_means_t2 = np.load('../../data/labels_classes_priors/prior_means_t2.npy')
prior_means = np.concatenate([prior_means_t1_hr, prior_means_t1_lr, prior_means_t2], axis=0)
prior_stds_t1_hr = np.load('../../data/labels_classes_priors/prior_stds_t1_hr.npy')
prior_stds_t1_lr = np.load('../../data/labels_classes_priors/prior_stds_t1_lr.npy')
prior_stds_t2 = np.load('../../data/labels_classes_priors/prior_stds_t2.npy')
prior_stds = np.concatenate([prior_stds_t1_hr, prior_stds_t1_lr, prior_stds_t2], axis=0)

# augmentation parameters
flipping = True
scaling_bounds = 0.1
rotation_bounds = 8
shearing_bounds = 0.01
translation_bounds = False
nonlin_std = 2.
bias_field_std = 0.2

# blurring/downsampling parameters
data_res = np.array([[1., 1., 3.], [1., 4.5, 1.]])  # slice spacing for the input channels only
thickness = np.array([[1., 1., 3.], [1., 3., 1.]])
randomise_res = False
downsample = True
build_reliability_maps = True
blur_range = 1.15
simulate_registration_error = True


########################################################################################################

# launch training
training(labels_folder,
         model_dir,
         prior_means,
         prior_stds,
         images_dir=images_folder,
         path_generation_labels=generation_labels,
         path_generation_classes=generation_classes,
         batchsize=1,
         input_channels=input_channels,
         output_channel=output_channel,
         target_res=target_res,
         output_shape=output_shape,
         flipping=flipping,
         scaling_bounds=scaling_bounds,
         rotation_bounds=rotation_bounds,
         shearing_bounds=shearing_bounds,
         translation_bounds=translation_bounds,
         nonlin_std=nonlin_std,
         simulate_registration_error=True,
         data_res=data_res,
         thickness=thickness,
         randomise_res=randomise_res,
         downsample=downsample,
         blur_range=blur_range,
         build_reliability_maps=build_reliability_maps,
         bias_field_std=bias_field_std,
         n_levels=n_levels,
         nb_conv_per_level=nb_conv_per_level,
         conv_size=conv_size,
         unet_feat_count=unet_feat_count,
         feat_multiplier=feat_multiplier,
         dropout=dropout,
         activation=activation,
         lr=learning_rate,
         lr_decay=lr_decay,
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         regression_metric=regression_metric,
         work_with_residual_channel=work_with_residual_channel)
