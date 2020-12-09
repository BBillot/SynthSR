# python imports
import numpy as np
import numpy.random as npr

# third-party imports
from ext.lab2im import utils


def build_model_inputs(path_label_maps,
                       n_labels,
                       prior_means,
                       prior_stds,
                       path_images=None,
                       batchsize=1,
                       n_channels=1,
                       generation_classes=None):
    """
    This function builds a generator to be fed to the lab2im model. It enables to generate all the required inputs,
    according to the operations performed in the model.
    :param path_label_maps: list of the paths of the input label maps.
    :param n_labels: number of labels in the input label maps.
    :param prior_means: hyperparameters controlling the prior distributions of the GMM *means*.
    Each mean of the GMM is sampled at each mini-batch from from a Gaussian prior with two hyperparameters (mean and std
    dev). Depending on the number of channels, prior_means can thus be:
    1) if n_channels=1: an array of shape (2, K), where K is the number of classes (K=len(generation_labels) if
    generation_classes is not given). The mean of the Gaussian distribution associated to class k in [0, ...K-1] is
    sampled at each mini-batch from N(prior_means[0,k], prior_means[1,k]).
    2) if n_channels>1: an array of shape (2*n_channels, K), where the i-th block of two rows (for i in
    [0, ... n_channels]) corresponds to the hypreparameters of channel i.
    3) the path to a numpy array corresponding to cases 1 or 2.
    :param prior_stds: same as prior_means but for the standard deviations of the GMM.
    :param path_images: optionally, corresponding image intensities (useful for regression)
    :param batchsize: (optional) numbers of images to generate per mini-batch. Default is 1.
    :param n_channels: (optional) number of channels to be synthetised. Default is 1.
    :param generation_classes: (optional) Indices regrouping generation labels into classes of same intensity
    distribution. Regouped labels will thus share the same Gaussian when samling a new image. Can be a sequence or a
    1d numpy array. It should have the same length as generation_labels, and contain values between 0 and K-1, where K
    is the total number of classes. Default is all labels have different classes.
    """

    # get label info
    _, _, n_dims, _, _, _ = utils.get_volume_info(path_label_maps[0])

    # allocate unique class to each label if generation classes is not given
    if generation_classes is None:
        generation_classes = np.arange(n_labels)

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = npr.randint(len(path_label_maps), size=batchsize)

        # initialise input lists
        list_label_maps = []
        list_means = []
        list_stds = []
        list_images = []

        for idx in indices:

            # add labels to inputs
            lab = utils.load_volume(path_label_maps[idx], dtype='int', aff_ref=np.eye(4))
            list_label_maps.append(utils.add_axis(lab, axis=-2))

            if path_images is not None:
                im = utils.load_volume(path_images[idx], dtype='float')
                list_images.append(im[np.newaxis, :, :, :, np.newaxis])

            # add means and standard deviations to inputs
            means = np.empty((n_labels, 0))
            stds = np.empty((n_labels, 0))
            for channel in range(n_channels):

                # retrieve channel specific stats if necessary
                if isinstance(prior_means, np.ndarray):
                    if prior_means.shape[0] / 2 != n_channels:
                        raise ValueError("the number of blocks in prior_means does not match n_channels.")
                    tmp_prior_means = prior_means[2 * channel:2 * channel + 2, :]
                else:
                    tmp_prior_means = prior_means
                if isinstance(prior_stds, np.ndarray):
                    if prior_stds.shape[0] / 2 != n_channels:
                        raise ValueError("the number of blocks in prior_stds does not match n_channels.")
                    tmp_prior_stds = prior_stds[2 * channel:2 * channel + 2, :]
                else:
                    tmp_prior_stds = prior_stds

                # draw means and std devs from priors
                tmp_classes_means = utils.draw_value_from_distribution(tmp_prior_means, n_labels, 'normal', 125., 100.,
                                                                       positive_only=True)
                tmp_classes_stds = utils.draw_value_from_distribution(tmp_prior_stds, n_labels, 'normal', 15., 10.,
                                                                      positive_only=True)
                tmp_means = utils.add_axis(tmp_classes_means[generation_classes], -1)
                tmp_stds = utils.add_axis(tmp_classes_stds[generation_classes], -1)
                means = np.concatenate([means, tmp_means], axis=1)
                stds = np.concatenate([stds, tmp_stds], axis=1)
            list_means.append(utils.add_axis(means))
            list_stds.append(utils.add_axis(stds))

        # build list of inputs of augmentation model
        list_inputs = [list_label_maps, list_means, list_stds]
        if path_images is not None:
            list_inputs.append(list_images)

        # concatenate individual input types if batchsize > 1
        if batchsize > 1:
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs


def means_stds_fs_labels_with_relations(means_range, std_devs_range, min_diff=15, head=True):

    # draw gm wm and csf means
    gm_wm_csf_means = np.zeros(3)
    while (abs(gm_wm_csf_means[1] - gm_wm_csf_means[0]) < min_diff) | \
          (abs(gm_wm_csf_means[1] - gm_wm_csf_means[2]) < min_diff) | \
          (abs(gm_wm_csf_means[0] - gm_wm_csf_means[2]) < min_diff):
        gm_wm_csf_means = utils.draw_value_from_distribution(means_range, 3, 'uniform', 125., 100., positive_only=True)
        gm_wm_csf_means = utils.add_axis(gm_wm_csf_means, -1)

    # apply relations
    wm = gm_wm_csf_means[0]
    gm = gm_wm_csf_means[1]
    csf = gm_wm_csf_means[2]
    csf_like = csf * npr.uniform(low=0.95, high=1.05)
    alpha_thalamus = npr.uniform(low=0.4, high=0.9)
    thalamus = alpha_thalamus*gm + (1-alpha_thalamus)*wm
    cerebellum_wm = wm * npr.uniform(low=0.7, high=1.3)
    cerebellum_gm = gm * npr.uniform(low=0.7, high=1.3)
    caudate = gm * npr.uniform(low=0.9, high=1.1)
    putamen = gm * npr.uniform(low=0.9, high=1.1)
    hippocampus = gm * npr.uniform(low=0.9, high=1.1)
    amygdala = gm * npr.uniform(low=0.9, high=1.1)
    accumbens = caudate * npr.uniform(low=0.9, high=1.1)
    pallidum = wm * npr.uniform(low=0.8, high=1.2)
    brainstem = wm * npr.uniform(low=0.8, high=1.2)
    alpha_ventralDC = npr.uniform(low=0.1, high=0.6)
    ventralDC = alpha_ventralDC*gm + (1-alpha_ventralDC)*wm
    alpha_choroid = npr.uniform(low=0.0, high=1.0)
    choroid = alpha_choroid*csf + (1-alpha_choroid)*wm

    # regroup structures
    neutral_means = [np.zeros(1), csf_like, csf_like, brainstem, csf]
    sided_means = [wm, gm, csf_like, csf_like, cerebellum_wm, cerebellum_gm, thalamus, caudate, putamen, pallidum,
                   hippocampus, amygdala, accumbens, ventralDC, choroid]

    # draw std deviations
    std = utils.draw_value_from_distribution(std_devs_range, 17, 'uniform', 15., 10., positive_only=True)
    std = utils.add_axis(std, -1)
    neutral_stds = [np.zeros(1), std[1], std[1], std[2], std[3]]
    sided_stds = [std[4], std[5], std[1], std[1], std[6], std[7], std[8], std[9], std[10], std[11], std[12], std[13],
                  std[14], std[15], std[16]]

    # add means and variances for extra head labels if necessary
    if head:
        # means
        extra_means = utils.draw_value_from_distribution(means_range, 2, 'uniform', 125., 100., positive_only=True)
        extra_means = utils.add_axis(extra_means, -1)
        skull = extra_means[0]
        soft_non_brain = extra_means[1]
        eye = csf * npr.uniform(low=0.95, high=1.05)
        optic_chiasm = wm * npr.uniform(low=0.8, high=1.2)
        vessel = csf * npr.uniform(low=0.7, high=1.3)
        neutral_means += [csf_like, optic_chiasm, skull, soft_non_brain, eye]
        sided_means.insert(-1, vessel)
        # std dev
        extra_std = utils.draw_value_from_distribution(std_devs_range, 4, 'uniform', 15., 10., positive_only=True)
        extra_std = utils.add_axis(extra_std, -1)
        neutral_stds += [std[1], extra_std[0], extra_std[1], extra_std[2], std[1]]
        sided_stds.insert(-1, extra_std[3])

    means = np.concatenate([np.array(neutral_means), np.array(sided_means), np.array(sided_means)])
    stds = np.concatenate([np.array(neutral_stds), np.array(sided_stds), np.array(sided_stds)])

    return means, stds
