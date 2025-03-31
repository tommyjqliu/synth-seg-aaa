"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers
from ext.lab2im import edit_tensors as l2i_et
from ext.lab2im.edit_volumes import get_ras_axes
from SynthSegAAA.perlin_noise_layer import PerlinNoise

def labels_to_image_model(labels_shape,
                          n_channels,
                          generation_labels,
                          output_labels,
                          n_neutral_labels,
                          atlas_res,
                          target_res,
                          output_shape=None,
                          output_div_by_n=None,
                          flipping=True,
                          aff=None,
                          scaling_bounds=0.2,
                          rotation_bounds=15,
                          shearing_bounds=0.012,
                          translation_bounds=False,
                          nonlin_std=3.,
                          nonlin_scale=.0625,
                          randomise_res=False,
                          max_res_iso=4.,
                          max_res_aniso=8.,
                          data_res=None,
                          thickness=None,
                          bias_field_std=.5,
                          bias_scale=.025,
                          return_gradients=False):

    # reformat resolutions
    labels_shape = utils.reformat_to_list(labels_shape)
    n_dims, _ = utils.get_dims(labels_shape)
    atlas_res = utils.reformat_to_n_channels_array(atlas_res, n_dims, n_channels)
    data_res = atlas_res if data_res is None else utils.reformat_to_n_channels_array(data_res, n_dims, n_channels)
    thickness = data_res if thickness is None else utils.reformat_to_n_channels_array(thickness, n_dims, n_channels)
    atlas_res = atlas_res[0]
    target_res = atlas_res if target_res is None else utils.reformat_to_n_channels_array(target_res, n_dims)[0]

    # get shapes
    crop_shape, output_shape = get_shapes(labels_shape, output_shape, atlas_res, target_res, output_div_by_n)

    # define model inputs
    labels_input = KL.Input(shape=labels_shape + [1], name='labels_input', dtype='int32')
    means_input = KL.Input(shape=list(generation_labels.shape) + [n_channels], name='means_input')
    stds_input = KL.Input(shape=list(generation_labels.shape) + [n_channels], name='std_devs_input')
    list_inputs = [labels_input, means_input, stds_input]

    # deform labels
    labels = layers.RandomSpatialDeformation(scaling_bounds=scaling_bounds,
                                             rotation_bounds=rotation_bounds,
                                             shearing_bounds=shearing_bounds,
                                             translation_bounds=translation_bounds,
                                             nonlin_std=nonlin_std,
                                             nonlin_scale=nonlin_scale,
                                             inter_method='nearest')(labels_input)

    # cropping
    if crop_shape != labels_shape:
        labels = layers.RandomCrop(crop_shape)(labels)

    # flipping
    if flipping:
        assert aff is not None, 'aff should not be None if flipping is True'
        labels = layers.RandomFlip(get_ras_axes(aff, n_dims)[0], True, generation_labels, n_neutral_labels)(labels)

    # build synthetic image
    image = layers.SampleConditionalGMM(generation_labels)([labels, means_input, stds_input])

    # apply bias field
    if bias_field_std > 0:
        image = layers.BiasFieldCorruption(bias_field_std, bias_scale, False)(image)

    # intensity augmentation
    image = layers.IntensityAugmentation(clip=300, normalise=True, gamma_std=.5, separate_channels=True)(image)

    image = PerlinNoise(scale=30, octaves=6, persistence=0.5, lacunarity=2.0, amplitude=1)([image, labels])

    # loop over channels
    channels = list()
    split = KL.Lambda(lambda x: tf.split(x, [1] * n_channels, axis=-1))(image) if (n_channels > 1) else [image]
    for i, channel in enumerate(split):

        if randomise_res:
            max_res_iso = np.array(utils.reformat_to_list(max_res_iso, length=n_dims, dtype='float'))
            max_res_aniso = np.array(utils.reformat_to_list(max_res_aniso, length=n_dims, dtype='float'))
            max_res = np.maximum(max_res_iso, max_res_aniso)
            resolution, blur_res = layers.SampleResolution(atlas_res, max_res_iso, max_res_aniso)(means_input)
            sigma = l2i_et.blurring_sigma_for_downsampling(atlas_res, resolution, thickness=blur_res)
            channel = layers.DynamicGaussianBlur(0.75 * max_res / np.array(atlas_res), 1.03)([channel, sigma])
            channel = layers.MimicAcquisition(atlas_res, atlas_res, output_shape, False)([channel, resolution])
            channels.append(channel)

        else:
            sigma = l2i_et.blurring_sigma_for_downsampling(atlas_res, data_res[i], thickness=thickness[i])
            channel = layers.GaussianBlur(sigma, 1.03)(channel)
            resolution = KL.Lambda(lambda x: tf.convert_to_tensor(data_res[i], dtype='float32'))([])
            channel = layers.MimicAcquisition(atlas_res, data_res[i], output_shape)([channel, resolution])
            channels.append(channel)

    # concatenate all channels back
    image = KL.Lambda(lambda x: tf.concat(x, -1))(channels) if len(channels) > 1 else channels[0]

    # compute image gradient
    if return_gradients:
        image = layers.ImageGradients('sobel', True, name='image_gradients')(image)
        image = layers.IntensityAugmentation(clip=10, normalise=True)(image)

    # resample labels at target resolution
    if crop_shape != output_shape:
        labels = l2i_et.resample_tensor(labels, output_shape, interp_method='nearest')

    # map generation labels to segmentation values
    labels = layers.ConvertLabels(generation_labels, dest_values=output_labels, name='labels_out')(labels)

    # build model (dummy layer enables to keep the labels when plugging this model to other models)
    image = KL.Lambda(lambda x: x[0], name='image_out')([image, labels])
    brain_model = Model(inputs=list_inputs, outputs=[image, labels])

    return brain_model


def get_shapes(labels_shape, output_shape, atlas_res, target_res, output_div_by_n):

    # reformat resolutions to lists
    atlas_res = utils.reformat_to_list(atlas_res)
    n_dims = len(atlas_res)
    target_res = utils.reformat_to_list(target_res)

    # get resampling factor
    if atlas_res != target_res:
        resample_factor = [atlas_res[i] / float(target_res[i]) for i in range(n_dims)]
    else:
        resample_factor = None

    # output shape specified, need to get cropping shape, and resample shape if necessary
    if output_shape is not None:
        output_shape = utils.reformat_to_list(output_shape, length=n_dims, dtype='int')

        # make sure that output shape is smaller or equal to label shape
        if resample_factor is not None:
            output_shape = [min(int(labels_shape[i] * resample_factor[i]), output_shape[i]) for i in range(n_dims)]
        else:
            output_shape = [min(labels_shape[i], output_shape[i]) for i in range(n_dims)]

        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            tmp_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in output_shape]
            if output_shape != tmp_shape:
                print('output shape {0} not divisible by {1}, changed to {2}'.format(output_shape, output_div_by_n,
                                                                                     tmp_shape))
                output_shape = tmp_shape

        # get cropping and resample shape
        if resample_factor is not None:
            cropping_shape = [int(np.around(output_shape[i]/resample_factor[i], 0)) for i in range(n_dims)]
        else:
            cropping_shape = output_shape

    # no output shape specified, so no cropping unless label_shape is not divisible by output_div_by_n
    else:

        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:

            # if resampling, get the potential output_shape and check if it is divisible by n
            if resample_factor is not None:
                output_shape = [int(labels_shape[i] * resample_factor[i]) for i in range(n_dims)]
                output_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in output_shape]
                cropping_shape = [int(np.around(output_shape[i] / resample_factor[i], 0)) for i in range(n_dims)]
            # if no resampling, simply check if image_shape is divisible by n
            else:
                cropping_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in labels_shape]
                output_shape = cropping_shape

        # if no need to be divisible by n, simply take cropping_shape as image_shape, and build output_shape
        else:
            cropping_shape = labels_shape
            if resample_factor is not None:
                output_shape = [int(cropping_shape[i] * resample_factor[i]) for i in range(n_dims)]
            else:
                output_shape = cropping_shape

    return cropping_shape, output_shape
