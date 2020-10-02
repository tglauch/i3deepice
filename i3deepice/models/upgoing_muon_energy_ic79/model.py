''' new Network definitions using the functional API from keras.
first part: model settings like input variables, outputs and transformations
second part: model definition, name must be def model(input_shape):
'''

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import sys
from collections import OrderedDict
import os
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path)
sys.path.append(os.path.join(this_path, ".."))
sys.path.append(os.path.join(this_path, '..', 'lib'))
import transformations as tr
import numpy as np
import block_units_tf as bunit



# *Settings*
# define inputs for each branch

def mask_definition(reco_vals):
    mask = (reco_vals["mu_e_on_entry"] < 50) | (reco_vals['classification'] == 11)
    return mask

inputs = OrderedDict()

inputs["Branch_IC_time"] = {"variables": ["IC_charge",'IC_time_first', 'IC_charge_10ns',
                                          'IC_charge_50ns', 'IC_charge_100ns',
                                          'IC_time_spread', 'IC_time_std', 'IC_time_weighted_median',
                                          'IC_pulse_0_01_pct_charge_quantile',
                                          'IC_pulse_0_03_pct_charge_quantile',
                                          'IC_pulse_0_05_pct_charge_quantile',
                                          'IC_pulse_0_11_pct_charge_quantile',
                                          'IC_pulse_0_15_pct_charge_quantile',
                                          'IC_pulse_0_2_pct_charge_quantile',
                                          'IC_pulse_0_5_pct_charge_quantile',
                                          'IC_pulse_0_8_pct_charge_quantile'],
                     "transformations": [tr.IC_divide_100, tr.IC_divide_10000, tr.IC_divide_100,  tr.IC_divide_100,
                                         tr.IC_divide_100, tr.IC_divide_10000,
                                         tr.IC_divide_10000, tr.IC_divide_10000, tr.IC_divide_10000,tr.IC_divide_10000,
                                         tr.IC_divide_10000, tr.IC_divide_10000, tr.IC_divide_10000, tr.IC_divide_10000,
                                         tr.IC_divide_10000, tr.IC_divide_10000]}


# define outputs for each branch
outputs = OrderedDict()
outputs["Out1"] = {"variables": ["e_on_entry"],
                   "transformations": [tr.log10]}
loss_weights = {'Target1': 1.}
loss_functions = ["mse"]
metrics = ['mae']
# Step 4: Define the model using Keras functional API
output_names = {0: 'mu_E_on_entry'}

def inception_block4(input_tensor, n, t0=2, t1=4, t2=5, n_pool=3, scale=0.1):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    tower_0 = bunit.conv3d_bn(input_tensor, n, (t0,1,1), padding='same')
    tower_0 = bunit.conv3d_bn(tower_0, n, (1,t0,1), padding='same')
    tower_0 = bunit.conv3d_bn(tower_0, n, (1,1,t0), padding='same')

    tower_1 = bunit.conv3d_bn(input_tensor, n, (t1,1,1), padding='same')
    tower_1 = bunit.conv3d_bn(tower_1, n, (1,t1,1), padding='same')
    tower_1 = bunit.conv3d_bn(tower_1, n, (1,1,t1), padding='same')

    tower_4 = bunit.conv3d_bn(input_tensor, n, (1,1,t2), padding='same')

    tower_3 = tensorflow.keras.layers.MaxPooling3D((n_pool, n_pool, n_pool),
                         	                   strides=(1,1,1), padding='same')(input_tensor)
    tower_3 = bunit.conv3d_bn(tower_3, n, (1,1,1), padding='same')

    up = tensorflow.keras.layers.concatenate(
        [tower_0, tower_1, tower_3, tower_4], axis = channel_axis)
    return up


def inception_resnet(input_tensor, n, t1=2, t2=3, n_pool=3, scale=0.1):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    tower_1 = bunit.conv3d_bn(input_tensor, n, (1,1,1), padding='same')
    tower_1 = bunit.conv3d_bn(tower_1, n, (t1,1,1), padding='same')
    tower_1 = bunit.conv3d_bn(tower_1, n, (1,t1,1), padding='same')
    tower_1 = bunit.conv3d_bn(tower_1, n, (1,1,t1), padding='same')

    tower_2 = bunit.conv3d_bn(input_tensor, n, (1,1,1), padding='same')
    tower_2 = bunit.conv3d_bn(tower_2, n, (t2,1,1), padding='same')
    tower_2 = bunit.conv3d_bn(tower_2, n, (1,t2,1), padding='same')
    tower_2 = bunit.conv3d_bn(tower_2, n, (1,1,t2), padding='same')

    tower_3 = tensorflow.keras.layers.MaxPooling3D((n_pool, n_pool, n_pool),
                                        strides=(1,1,1), padding='same')(input_tensor)
    tower_3 = bunit.conv3d_bn(tower_3, n, (1,1,1), padding='same')

    up = tensorflow.keras.layers.concatenate(
        [tower_1, tower_2, tower_3], axis = channel_axis)

    x = tensorflow.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(input_tensor)[1:],
               arguments={'scale': scale},)([input_tensor, up])

    return x

def model(input_shapes, output_shapes):
    ### Most important: Define your model using the functional API of Keras
    # https://keras.io/getting-started/functional-api-guide/

    # The Input
    input_b1 = tensorflow.keras.layers.Input(
        shape=input_shapes['Branch_IC_time']['general'],
        name = "Input-Branch1")
        

    # Hidden Layers
    z1 = inception_block4(input_b1, 18, t0=1, t1=4, t2=6)
    z1 = inception_block4(input_b1, 18, t0=2, t1=5, t2=8)
    z1 = inception_block4(z1, 18, t0=2, t1=3, t2=7)
    z1 = inception_block4(z1, 18, t0=2, t1=4, t2=8)
    z1 = inception_block4(z1, 18, t0=3, t1=5, t2=9)
    z1 = inception_block4(z1, 18, t0=3, t1=8, t2=9)
    z1 = tensorflow.keras.layers.AveragePooling3D(pool_size=(2, 2, 3))(z1)
    z1 = tensorflow.keras.layers.BatchNormalization()(z1)
    for i in range(6):
        z1 = inception_resnet(z1, 24, t2=3)
        z1 = inception_resnet(z1, 24, t2=4)
        z1 = inception_resnet(z1, 24, t2=5)
    z1 = tensorflow.keras.layers.AveragePooling3D(pool_size=(1, 1, 2))(z1)
    z1 = tensorflow.keras.layers.BatchNormalization()(z1)
    for i in range(6):
        z1 = inception_resnet(z1, 24, t2=3)
        z1 = inception_resnet(z1, 24, t2=4)
        z1 = inception_resnet(z1, 24, t2=5)
    z1 = tensorflow.keras.layers.Conv3D(64, (1, 1, 1), activation='relu',
            padding="same", name='conv1x1x1')(z1)
    z1 = tensorflow.keras.layers.Conv3D(4, (1, 1, 1), activation='relu',
            padding="same", name='conv1x1x1_2')(z1)
    z1 = tensorflow.keras.layers.AveragePooling3D(pool_size=(1, 1, 2))(z1)
    z1 = tensorflow.keras.layers.Flatten(data_format=K.image_data_format())(z1)
    z1 = tensorflow.keras.layers.Dense(120, name='dense1')(z1)
    z1 = tensorflow.keras.layers.Dense(64, name='dense2')(z1)
    z1 = tensorflow.keras.layers.Dense(16, name='dense3')(z1)
    print('out shape {}'.format(output_shapes["Out1"]["general"][0]))
    output_b1 = tensorflow.keras.layers.Dense(output_shapes["Out1"]["general"][0],
                                   activation="linear",
                                   name="Target1")(z1)    
    # The Output
    model= keras.models.Model(inputs=[input_b1],
                                outputs=[output_b1])
    print(model.summary())
    return model

