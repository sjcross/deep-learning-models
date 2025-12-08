from __future__ import print_function, unicode_literals, absolute_import, division

from stardist.models import Config2D, StarDist2D

conf = Config2D(n_channel_in=3, train_batch_size=4, train_shape_completion=False)
model = StarDist2D(conf, name='stardist_no_shape_completion', basedir='models')
model.load_weights("./weights_best.h5")
model.export_TF("./weights_best.zip")
