"""This scripts generates 3 examples for plain SR between LR synthetic T1 scans (input channel) and HR synthetic T1
scans (regression targets, or output channel). Therefore this script does not use real images as regerssion targets.
Because we do straight SR, the input channel is generated directly by blurring/downsampling the HR target regression
target, which are not corrupted nor downsampled in any way.

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
result_dir = '../../data/generated_images/2-SR_synthetic'  # folder where they will be saved

# general parameters
# Here we still synthesise 1 channel, whose HR version will serve as regression target, and LR version as input channel.
# So we set this channel to be both an input (with input_channels), and an output (with output_channel).
input_channels = True
output_channel = 0
# In this example we want to generate data at a lower resolution than the training label maps (this is obviously not
# very useful here, but it could be if you use training label maps at ultra high resolution, e.g. 0.3mm, and wish to
# regress at say 0.7mm). When providing a target resolution, both the input and output channels will be resampled to
# this resolution.
target_res = 1.5  # in mm
# In order to introduce variations in traninig examples (and to some extend model translation), we randomly crop the
# generated pairs to a given shape. If target_res is not None, this happens after resampling to target_resolution, so
# please provide a value bearing in mind that cropping is done at the provided target_res.
output_shape = 128

# label values of structure to generate from
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
# classes associating similar structures to the same Gaussian distribution
generation_classes = '../../data/labels_classes_priors/generation_classes.npy'

# We specify the hyperparameters both the synthesise channel.
prior_means = np.load('../../data/labels_classes_priors/prior_means_t1_lr.npy')
prior_stds = np.load('../../data/labels_classes_priors/prior_stds_t1_lr.npy')

# blurring/downsampling parameters
# We specify here the slice spacing/thickness that we want the input channel to mimic. We emphasise that none of these
# variables will affect the regression target, which is not blurred/downsampled to mimic the acquisition resolution.
data_res = np.array([1., 1., 3.])  # slice spacing i.e. resolution to mimic
thickness = np.array([1., 1., 3.])  # slice thickness
# Here we downsample the input channel to simulated LR (recommanded as the gap is large between LR and HR). We emphasise
# that the dowsampled scans are now going to be resampled to target_res, such that they voxels are perfectly aligned
# with the regression target.
downsample = True  #
build_reliability_maps = True  # add reliability map to input channels


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
                                 data_res=data_res,
                                 thickness=thickness,
                                 downsample=downsample,
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
                      os.path.join(result_dir, 'reliability_map_input_%s.nii.gz' % (n + 1)))
    utils.save_volume(np.squeeze(regression_target), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 't1_synthetic_target_%s.nii.gz' % (n + 1)))
