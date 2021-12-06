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


from argparse import ArgumentParser
from SynthSR.training import training
from ext.lab2im.utils import infer

parser = ArgumentParser()

# ------------------------------------------------- General parameters -------------------------------------------------
# Positional arguments
parser.add_argument("labels_dir", type=str)
parser.add_argument("model_dir", type=str)
parser.add_argument("prior_means", type=str)
parser.add_argument("prior_stds", type=str)
parser.add_argument("path_generation_labels", type=str)

# ---------------------------------------------- Generation parameters ----------------------------------------------
# training inputs parameters
parser.add_argument("--images", type=str, dest="images_dir", default=None)
parser.add_argument("--generation_classes", type=str, dest="path_generation_classes", default=None)
parser.add_argument("--prior_distributions", type=str, dest="prior_distributions", default='normal')

# output-related parameters
parser.add_argument("--batchsize", type=int, dest="batchsize", default=1)
parser.add_argument("--input_channels", type=str, dest="input_channels", default=True)
parser.add_argument("--output_channel", type=int, dest="output_channel", default=None)
parser.add_argument("--target_res", type=float, dest="target_res", default=None)
parser.add_argument("--output_shape", type=int, dest="output_shape", default=None)

# spatial deformation parameters
parser.add_argument("--no_flipping", action='store_false', dest="flipping")
parser.add_argument("--scaling", dest="scaling_bounds", type=infer, default=0.15)
parser.add_argument("--rotation", dest="rotation_bounds", type=infer, default=15)
parser.add_argument("--shearing", dest="shearing_bounds", type=infer, default=.02)
parser.add_argument("--translation", dest="translation_bounds", type=infer, default=5)
parser.add_argument("--nonlin_std", type=float, dest="nonlin_std", default=4.)
parser.add_argument("--nonlin_shape_factor", type=float, dest="nonlin_shape_factor", default=.03125)
parser.add_argument("--no_reg_error", action='store_false', dest="simulate_registration_error")

# blurring/resampling parameters
parser.add_argument("--data_res", dest="data_res", type=infer, default=None)
parser.add_argument("--thickness", dest="thickness", type=infer, default=None)
parser.add_argument("--downsample", action='store_true', dest="downsample")
parser.add_argument("--blur_range", type=float, dest="blur_range", default=1.15)
parser.add_argument("--no_rel_map", action='store_false', dest="build_reliability_maps")

# bias field parameters
parser.add_argument("--bias_std", type=float, dest="bias_field_std", default=.3)
parser.add_argument("--bias_shape_factor", type=float, dest="bias_shape_factor", default=.03125)

# -------------------------------------------- UNet architecture parameters --------------------------------------------
parser.add_argument("--n_levels", type=int, dest="n_levels", default=5)
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=2)
parser.add_argument("--conv_size", type=int, dest="conv_size", default=3)
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=24)
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2)
parser.add_argument("--dropout", type=float, dest="dropout", default=0.)
parser.add_argument("--activation", type=str, dest="activation", default='elu')

# ------------------------------------------------- Training parameters ------------------------------------------------
parser.add_argument("--lr", type=float, dest="lr", default=1e-4)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", default=0)
parser.add_argument("--epochs", type=int, dest="epochs", default=100)
parser.add_argument("--steps_per_epoch", type=int, dest="steps_per_epoch", default=1000)
parser.add_argument("--metric", type=str, dest="regression_metric", default='l1')
parser.add_argument("--residual_channel", type=int, dest="work_with_residual_channel", default=None)
parser.add_argument("--loss_cropping", type=int, dest="loss_cropping")
parser.add_argument("--checkpoint", type=str, dest="checkpoint", default=None)

# ------------------------------------------------- Support for regularization with segmentation -----------------------

parser.add_argument("--seg_reg_model_file", type=str, dest="segmentation_model_file", default=None)
parser.add_argument("--seg_reg_label_list", type=str, dest="segmentation_label_list", default=None)
parser.add_argument("--seg_reg_leabel_equiv", type=str, dest="segmentation_label_equivalency", default=None)
parser.add_argument("--seg_reg_rel_weight", type=float, dest="relative_weight_segmentation", default=0.25)


args = parser.parse_args()
training(**vars(args))
