"""This scripts generates 3 examples where we regress HR real scans from multi-modal LR scans. Specifically we regress
HR T1 scans from LR T1 and T2 scans. We assume here that HR label maps are available with corresponding T1 scans. Thus
this script produces pairs of real HR T1 scans along with aligned HR synthetic scans (input channels) simulating T1 and
T2 scans acquired at LR.


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
# folder containing corresponding images, that will be used as target regression
images_folder = '../../data/images/'

# result parameters
n_examples = 3  # number of generated examples
result_dir = '../../data/generated_images/5-SR-synthesis_real'  # folder where they will be saved

# general parameters
# We now generate 2 synthetic channels, which will both be used as input. Note that it only contains True values, since
# we use real scans as regeression target. Bear in mind that input_channels onyl refers to synthetic channels (it never
# includes the real regression target).
input_channels = [True, True]
output_channel = None  # the regression targets are not synthetic, but real
target_res = None  # produce data at the resolution of the label maps
output_shape = 128  # randomly crop to 128^3

# label values of structure to generate from
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
# classes associating similar structures to the same Gaussian distribution
generation_classes = '../../data/labels_classes_priors/generation_classes.npy'

# Hyperparameters governing the GMM priors for the synthetic T1 and T2 scans. Note that T1s will be the the first
# synthetic channel (as we provide t1 hyperparameters first).
prior_means_t1_lr = np.load('../../data/labels_classes_priors/prior_means_t1_lr.npy')
prior_means_t2 = np.load('../../data/labels_classes_priors/prior_means_t2.npy')
prior_means = np.concatenate([prior_means_t1_lr, prior_means_t2], axis=0)
prior_stds_t1_lr = np.load('../../data/labels_classes_priors/prior_stds_t1_lr.npy')
prior_stds_t2 = np.load('../../data/labels_classes_priors/prior_stds_t2.npy')
prior_stds = np.concatenate([prior_stds_t1_lr, prior_stds_t2], axis=0)

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
# corresponding resolution in the same order as for the hyperparameters. In this example we simulate:
# 3mm coronal T1 with 3mm thickness, and 4mm sagittal T2 with 3mm thickness.
data_res = np.array([[1., 1., 3.], [1., 4.5, 1.]])  # slice spacing
thickness = np.array([[1., 1., 3.], [1., 3., 1.]])  # slice thickness
downsample = True  # downsample to simulated LR
build_reliability_maps = True  # add reliability map to input channels
# In this example we introduce small variations in the blurring kernel, such that the downstream network is robust to
# small changes in acquisition resolution. We provide it here with this coefficient, where the blurring simulates a
# resolution sampled in the uniform distribution U(data_res/blur_range; data_res*blur_range). Therefore blur_range must
# equal to 1 (no changes), or greater than 1.
blur_range = 1.15
# Here we have two input channels, and we want to model registration problems between the two. This may be due to head
# movement between the two acquisitions, or the fact that the two scans were not acquired in the same coordinate space
# (e.g. orthogonal T1, and T2 acquired along the hippocampal axis). This registration error will be simulated with
# respect to the first input channel.
simulate_registration_error = True


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
                      os.path.join(result_dir, 't1_target_%s.nii.gz' % (n + 1)))
