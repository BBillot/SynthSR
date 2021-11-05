"""This scripts generates 3 examples where we now use multi-modal data to regress HR synthetic scans. Specifically we
regress HR synthetic T1 scans from LR T1 and T2 scans. Thus this script produces synthetic HR T1 scans as regression
target, and aligned HR synthetic scans (input channels) simulating T1 and T2 scans acquired at LR.


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
# We assume here that th HR and LR T1 scans have different contrasts, thus we generate 3 synthetic channels, the first
# one being the HR T1 scans, and the other being the LR T1 and T2 scans.
input_channels = [False, True, True]
output_channel = 0   # index corresponding to the regression target
target_res = None  # produce data at the resolution of the label maps
output_shape = 128  # randomly crop to 128^3
# As an additional example, if we were to regress HR T1 scans from LR T1 scans of the same contrast and T2 LR scans we
# would have synthesised 2 channels only, where the LR scans would have been derived from the HR ones with
# input_channels = [True, True], and output_channel = 0.

# label values of structure to generate from
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
# classes associating similar structures to the same Gaussian distribution
generation_classes = '../../data/labels_classes_priors/generation_classes.npy'

# Hyperparameters governing the GMM priors for the synthetic T1 and T2 scans.
# Following the order of input_channels, we first specify the hyperparameters of the HR T1 scans. The order of the input
# channels will then be given by the order in which we provide their corresponding hyperparameters. (LR T1 first, LR T2
# second).
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
# We assume here that the T1 and T2 LR scans were not acquired at the same resolution/slice thickness. We provide the
# corresponding resolution in the same order as for the GMM hyperparameters. In this example we simulate:
# 3mm coronal T1 with 3mm thickness, and 4mm sagittal T2 with 3mm thickness. Bear in mind that we only provide entries
# for input channels, so we do not specify anything for the first synthetic channel (output only).
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
                                 input_channels=input_channels,
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
    print('generation {0:d} took {1:.01f}s'.format(n + 1, end - start))

    # save output image and label map
    utils.save_volume(np.squeeze(input_channels[..., 0]), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 't1_input_%s.nii.gz' % (n + 1)))
    utils.save_volume(np.squeeze(input_channels[..., 1]), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'reliability_map_t1_input_%s.nii.gz' % (n + 1)))
    utils.save_volume(np.squeeze(input_channels[..., 2]), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 't2_input_%s.nii.gz' % (n + 1)))
    utils.save_volume(np.squeeze(input_channels[..., 3]), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'reliability_map_t2_input_%s.nii.gz' % (n + 1)))
    utils.save_volume(np.squeeze(regression_target), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 't1_synthetic_target_%s.nii.gz' % (n + 1)))
