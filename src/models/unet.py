import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

def down_block(x, filters, k=(3,3), pad="same", strides=1):
	c = layers.Conv2D(filters, k, padding=pad, strides=strides, activation="tanh")(x)
	c = layers.Conv2D(filters, k, padding=pad, strides=strides, activation="tanh")(c)
	p = layers.MaxPool2D((2,2), (2,2))(c)
	return c,p

def up_block(x, skip, filters, k=(3,3), pad="same", strides=1):
	us = layers.UpSampling2D((2,2))(x)
	con = layers.Concatenate()([us, skip])
	c = layers.Conv2D(filters, k, padding=pad, strides=strides, activation="tanh")(con)
	c = layers.Conv2D(filters, k, padding=pad, strides=strides, activation="tanh")(c)
	return c

def bottleneck(x, filters, k=(3,3), pad="same", strides=1):
	c = layers.Conv2D(filters, k, padding=pad, strides=strides, activation="tanh")(x)
	c = layers.Conv2D(filters, k, padding=pad, strides=strides, activation="tanh")(c)
	return c

def class_weighted_pixelwise_crossentropy(target, output):
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.1, 0.9]
    return -tf.reduce_sum(target * weights * tf.log(output))

def UNetModel(image_width,image_height,num_classes=1):
	f = [16, 32, 64, 128, 256]
	inputs = layers.Input((image_width, image_height, 1))

	p0 = inputs
	c1, p1 = down_block(p0, f[0])
	c2, p2 = down_block(p1, f[1])
	c3, p3 = down_block(p2, f[2])
	c4, p4 = down_block(p3, f[3])

	bn = bottleneck(p4,f[4])

	u1 = up_block(bn, c4, f[3])
	u2 = up_block(u1, c3, f[2])
	u3 = up_block(u2, c2, f[1])
	u4 = up_block(u3, c1, f[0])

	t1 = layers.Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(u4)
	# outputs = layers.Reshape((image_width*image_height,num_classes))(t1)
	
	model = models.Model(inputs, t1)

	return model
