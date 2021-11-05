"""Examples to show how to estimate of the hyperparameters governing the GMM prior distributions.
This requires images of the desired contrasts along with corresponding label maps (possibly obtained with automated
segmentation methods)


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


from SynthSR.estimate_priors import build_intensity_stats

# ----------------------------------------------- simple uni-modal case ------------------------------------------------

# paths of directories containing the images and corresponding label maps
image_dir = '../../data/images/'
labels_dir = '../../data/labels'
# list of labels from which we want to evaluate the GMM prior distributions
estimation_labels = '../data/labels_classes_priors/generation_labels.npy'
# path of folder where to write estimated priors
result_dir = '../data/estimated_priors'

build_intensity_stats(list_image_dir=image_dir,
                      list_labels_dir=labels_dir,
                      estimation_labels=estimation_labels,
                      result_dir=result_dir,
                      rescale=True)

# ------------------------------------ building Gaussian priors from several labels ------------------------------------

# same as before
image_dir = '../../data/images/'
labels_dir = '../../data/labels'
estimation_labels = '../data/labels_classes_priors/generation_labels.npy'
result_dir = '../data/estimated_priors'

# In the previous example, each label value is used to build the priors of a single Gaussian distribution.
# We show here how to build Gaussian priors from intensities associated with several label values.
# This is done by specifying a vector, which regroups label values into "classes". This must be a list of the same
# length as generation_labels, indicating the class of each label. Importantly the class values must be between 0 and
# K-1, where K is the total number of different classes. Labels sharing the same class will contribute to the
# construction of the same Gaussian prior.
# Example: generation_labels = [0, 259, 2, 3, 17]
#         generation_classes = [0,   3, 1, 2,  2]
# Here the intensities of labels 3 and 17 will contribute to building the Gaussian distribution #2.
estimation_classes = '../data/labels_classes_priors/generation_classes.npy'

build_intensity_stats(list_image_dir=image_dir,
                      list_labels_dir=labels_dir,
                      estimation_labels=estimation_labels,
                      estimation_classes=estimation_classes,
                      result_dir=result_dir,
                      rescale=True)

# -------------------------------------  multi-modal images with separate channels -------------------------------------

# Here we have multi-modal images, where the different channels are stored in separate directories.
# We provide these directories as a list.
list_image_dir = ['../data/images/images_t1', '../data/images/images_t2']
# In this example, we assume that channels are registered and at the same resolutions.
# Therefore we can use the same label maps for all channels.
labels_dir = '../../data/labels'

# same as before
estimation_labels = '../../data/labels_classes_priors/generation_labels.npy'
estimation_classes = '../../data/labels_classes_priors/generation_classes.npy'
result_dir = '../../data/estimated_PV-SynthSeg_priors'

# we do not provide the data for this example
# build_intensity_stats(list_image_dir=list_image_dir,
#                       list_labels_dir=labels_dir,
#                       estimation_labels=estimation_labels,
#                       estimation_classes=estimation_classes,
#                       result_dir=result_dir,
#                       rescale=True)

# ------------------------------------ multi-modal case with unregistered channels -------------------------------------

# Again, we have multi-modal images where the different channels are stored in separate directories.
list_image_dir = ['../data/images/images_t1', '../data/images/images_t2']
# In this example, we assume that the channels are no longer registered.
# Therefore we cannot use the same label maps for all channels, and must provide label maps for all modalities.
labels_dir = ['../data/images/labels_t1', '../data/images/labels_t2']

# same as before
estimation_labels = '../../data/labels_classes_priors/generation_labels.npy'
estimation_classes = '../../data/labels_classes_priors/generation_classes.npy'
result_dir = '../../data/estimated_PV-SynthSeg_priors'

# Again, we do not provide the data to examplify this case
# build_intensity_stats(list_image_dir=list_image_dir,
#                       list_labels_dir=labels_dir,
#                       estimation_labels=estimation_labels,
#                       estimation_classes=estimation_classes,
#                       result_dir=result_dir,
#                       rescale=True)
