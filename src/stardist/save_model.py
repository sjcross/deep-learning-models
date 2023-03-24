from __future__ import print_function, unicode_literals, absolute_import, division

from stardist.models import Config2D, StarDist2D

import os
import numpy as np

np.random.seed(42)
path = "C:\\Users\\sc13967\\Desktop\\2023-01-26 Matt Butler StarDist training\\"
batch_size = 1
image_width = 320

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

conf = Config2D(n_channel_in=1, train_batch_size=4, train_shape_completion=False)
model = StarDist2D(conf, name='stardist_no_shape_completion', basedir='models')
model.load_weights("F:\\Python Projects\\deep-learning-models\\models\\stardist_no_shape_completion\\weights_now.h5")
model.export_TF("C:\\Users\\sc13967\\Desktop\\weights_best.zip")
