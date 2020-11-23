from keras.layers import Input, Dense, Conv2D, concatenate
from keras.models import Model
from scipy.misc import imread, imsave, imresize
import numpy as np
from utils import *
import tensorflow as tf
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

# set paths
train_path = 'E:/xuben/train2014'
model_save_path = './models'

# get train images
train_list = list_images(train_path)
train_list = train_list[:-1]
np.random.shuffle(train_list)
train_images = get_train_images(train_list)

# define models
inputs = Input(shape=(256, 256, 1))
c1 = Conv2D(filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros')

dc1 = Conv2D(filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros')

dc2 = Conv2D(filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros')

dc3 = Conv2D(filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros')

tense1 = c1(inputs)
tense2 = dc1(tense1)
tense3 = dc2(concatenate([tense1, tense2], axis=-1))
tense4 = dc3(concatenate([tense1, tense2, tense3], axis=-1))
image_feature = concatenate([tense1, tense2, tense3, tense4], axis=-1)

c2 = Conv2D(filters=64,
             kernel_size=(3, 3),
             strides=(1, 1),
             padding='same',
             activation='relu',
             use_bias=True,
             kernel_initializer='glorot_uniform',
             bias_initializer='zeros')

c3 = Conv2D(filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros')

c4 = Conv2D(filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros')

c5 = Conv2D(filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros')

outputs = c5(c4(c3(c2(image_feature))))
model = Model(inputs=inputs, outputs=outputs)

# start training
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])
model.fit(x=train_images, y=train_images, batch_size=2, epochs=4)
model.save(model_save_path + '/mymodel.h5')

print('ok')