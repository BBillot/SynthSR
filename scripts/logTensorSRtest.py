"""This script trains a SR model for diffusion MRI (log-tensors) using also structural T1 and T2 as inputs."""



import numpy as np
import sys
from SynthSR.training import training
import os

sys.path.append("/autofs/homes/002/iglesias/python/code/SynthSR")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
if False:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"


# we have to specify a model dir, where the models will be saved after each epoch
model_dir = '/cluster/scratch/friday/models/diffusion/'  # folder where they will be saved

# general
regression_metric = 'l1'  # metric used to compute the loss function
labels_folder = '/autofs/space/panamint_005/users/iglesias/data/ImputationSynthesis/DTI_SR/data/label_maps'
images_folder = None
target_res = None # will use 0.7 from the label maps
output_shape = 64 # [size of label maps is 100x95x94]
loss_cropping = 56

# channels
input_channels = [True, True, True, True, True, True, True, True]  # specify which channel will be used as input channel for the network
simulate_registration_error = [False,False,False,False,False,False,True,True]


# GMM-sampling parameters
generation_labels = '/autofs/space/panamint_005/users/iglesias/data/ImputationSynthesis/DTI_SR/data/stats_files/generation_labels.npy'
generation_classes = None
prior_means = np.load('/autofs/space/panamint_005/users/iglesias/data/ImputationSynthesis/DTI_SR/data/stats_files/means_for_l1.npy')
prior_stds = np.load('/autofs/space/panamint_005/users/iglesias/data/ImputationSynthesis/DTI_SR/data/stats_files/stds_for_l1.npy')
prior_stds = prior_stds * 0.5;

# augmentation parameters
scaling_bounds = 0.1
rotation_bounds = 5 # keep small
shearing_bounds = 0.01
translation_bounds = False
nonlin_std = 2.0

# blurring/downsampling parameters
data_res = np.array([[2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
thickness = np.array([[2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
downsample = True
build_reliability_maps = False
blur_range = 1.15

output_channel = [0, 1, 2, 3, 4, 5]   # index corresponding to the regression target
work_with_residual_channel = [0, 1, 2, 3, 4, 5]


####
#  I shouldn't need to touch these
####

# training data
FS_sort = False

# Augmentation
flipping = False # no symmetry
bias_field_std = 0.0 # no bias in diffusion parameters

#  Unet architecture
n_levels = 5
nb_conv_per_level = 2
conv_size = 3
unet_feat_count = 24
feat_multiplier = 2
dropout = 0
activation = 'elu'

# learning parameters
learning_rate = 1e-4  # learning rate to apply
lr_decay = 0
epochs = 2000  # number of epochs
steps_per_epoch = 500  # number of steps per epoch



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
	 loss_cropping=loss_cropping,
         flipping=flipping,
         scaling_bounds=scaling_bounds,
         rotation_bounds=rotation_bounds,
         shearing_bounds=shearing_bounds,
         translation_bounds=translation_bounds,
         nonlin_std=nonlin_std,
         simulate_registration_error=simulate_registration_error,
         data_res=data_res,
         thickness=thickness,
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
         learning_rate=learning_rate,
         lr_decay=lr_decay,
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         regression_metric=regression_metric,
         work_with_residual_channel=work_with_residual_channel,
         FS_sort=FS_sort)
