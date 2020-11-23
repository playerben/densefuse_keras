from keras.layers import Input, Dense, Conv2D, concatenate
from keras.models import Model, load_model
from keras import backend
from scipy.misc import imread, imsave, imresize
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

# set paths
ir_path = './images/1a.bmp'
vis_path = './images/1b.bmp'
model_path = './models/mymodel.h5'
save_path = './outputs/'

# load data
ir_img = imread(ir_path)
vis_img = imread(vis_path)
ir_img = imresize(ir_img, [256, 256], interp='nearest')
ir_img = ir_img.reshape([1, 256, 256, 1])
vis_img = imresize(vis_img, [256, 256], interp='nearest')
vis_img = vis_img.reshape([1, 256, 256, 1])
model = load_model(model_path)

# start test
encoder = backend.function([model.layers[0].input], [model.layers[7].output])
decoder = backend.function([model.layers[8].input], [model.layers[11].output])

ir_feature = encoder([ir_img])[0]
vis_feature = encoder([vis_img])[0]
fused_feature = ir_feature + vis_feature
fused_img = decoder([fused_feature])[0]
fused_img = fused_img.reshape([256, 256])

# show
# for i in range(64):
#             show_img = fused_feature[:, :, :, i]
#             show_img.shape = [256, 256]
#             plt.subplot(8, 8, i + 1)
#             plt.imshow(show_img, cmap='gray')
#             plt.axis('off')
plt.imshow(fused_img, cmap='gray')
plt.axis('off')
plt.show()
print('ok')