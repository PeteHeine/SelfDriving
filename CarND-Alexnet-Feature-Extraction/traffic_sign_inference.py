"""
The traffic signs are 32x32 so you
have to resize them to be 227x227 before
passing them to AlexNet.
"""
import time
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from caffe_classes import class_names
from alexnet import AlexNet
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

#with tf.device('/gpu:0'):
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# TODO: Resize the images so they can be fed into AlexNet.
# HINT: Use `tf.image.resize_images` to resize the images
resized = tf.image.resize_images(x,(227,227))
probs = AlexNet(resized)
#device_count = {"GPU": 0}
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

init = tf.global_variables_initializer()
with tf.Session(config=config) as sess:
    sess.run(init)
    # Run Inference
    t = time.time()
    output = sess.run(probs, feed_dict={x: [im1, im2]})



# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
