"""This scripts generates 3 examples where we regress HR T1 scans from LR T1 and T2 scans. We assume here that no HR
images are available with the training label maps. Thus this script produces synthetic HR T1 scans as regression
target, and aligned HR synthetic scans (input channels) simulating T1 and T2 scans acquired at LR."""

import os
import time
import numpy as np
from ext.lab2im import utils
from SynthSR.brain_generator import BrainGenerator


# folder containing label maps to generate images from
labels_folder = '../../data/labels'
# no real images are used in this case
images_folder = None

# result parameters
n_examples = 3  # number of generated examples
result_dir = '../../data/generated_images/6-SR-synthesis_synthetic'  # folder where they will be saved

# general parameters
channels = 3  # we generate 3 synthetic channel (HR T1, LR T1 and LR T2).
output_channel = 0   # index corresponding to the regression target
target_res = None  # produce data at the resolution of the label maps
output_shape = 128  # randomly crop to 128^3

# label values of structure to generate from
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
# classes associating similar structures to the same Gaussian distribution
generation_classes = '../../data/labels_classes_priors/generation_classes.npy'

# Hyperparameters governing the GMM priors for the synthetic T1 and T2 scans.
# Note that we provide HR T1 hyperparameters first (because output_channel = 0). Following the order in which we
# specified the other hyperparameters, the synthetic T1s will be the the first
# input channel (as we provide t1 hyperparameters first), and T2 will come second.
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
scaling_bounds = 0.7
rotation_bounds = 8
shearing_bounds = 0.01
translation_bounds = False
nonlin_std = 2.
bias_field_std = 0.2

# blurring/downsampling parameters
# We assume here that the T1 and T2 LR scans were not acquired at the same resolution/slice thickness. We provide the
# corresponding resolution in the same order as for the hyperparameters. In this example we simulate:
# 3mm coronal T1 with 3mm thickness, and 4mm sagittal T2 with 3mm thickness. Note that we do not provide entries for
# the regression target as it will not be downsampled.
data_res = np.array([[1., 1., 3.], [1., 4.5, 1.]])  # slice spacing
thickness = np.array([[1., 1., 3.], [1., 3., 1.]])  # slice thickness
downsample = True  # downsample to simulated LR
build_reliability_maps = True  # add reliability map to input channels
blur_range = 1.15  # randomise blurring kernel
simulate_registration_error = True  # simulate registration mistakes between the synthetic input channels.


########################################################################################################

# instantiate BrainGenerator object
brain_generator = BrainGenerator(labels_dir=labels_folder,
                                 images_dir=images_folder,
                                 generation_labels=generation_labels,
                                 n_channels=channels,
                                 output_channel=output_channel,
                                 target_res=target_res,
                                 output_shape=output_shape,
                                 generation_classes=generation_classes,
                                 prior_means=prior_means,
                                 prior_stds=prior_stds,
                                 flipping=flipping,
                                 scaling_bounds=scaling_bounds,
                                 rotation_bounds=rotation_bounds,
                                 shearing_bounds=shearing_bounds,
                                 translation_bounds=translation_bounds,
                                 simulate_registration_error=simulate_registration_error,
                                 nonlin_std=nonlin_std,
                                 bias_field_std=bias_field_std,
                                 data_res=data_res,
                                 thickness=thickness,
                                 downsample=downsample,
                                 blur_range=blur_range,
                                 build_reliability_maps=build_reliability_maps)

# create result dir
utils.mkdir(result_dir)

for n in range(n_examples):

    # generate !
    start = time.time()
    input_channels, regression_target = brain_generator.generate_brain()
    end = time.time()
    print('generation {0:d} took {1:.01f}s'.format(n+1, end - start))

    # save output image and label map
    utils.save_volume(np.squeeze(input_channels[..., 0]), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 't1_input_%s.nii.gz' % n))
    utils.save_volume(np.squeeze(input_channels[..., 1]), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'reliability_map_t1_input_%s.nii.gz' % n))
    utils.save_volume(np.squeeze(input_channels[..., 2]), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 't2_input_%s.nii.gz' % n))
    utils.save_volume(np.squeeze(input_channels[..., 3]), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'reliability_map_t2_input_%s.nii.gz' % n))
    utils.save_volume(np.squeeze(regression_target), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 't1_synthetic_target_%s.nii.gz' % n))
