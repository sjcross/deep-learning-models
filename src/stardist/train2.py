from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from tensorflow.keras.callbacks import ModelCheckpoint

from stardist import fill_label_holes, random_label_cmap
from stardist.models import Config2D, StarDist2D, StarDistData2D

np.random.seed(42)
lbl_cmap = random_label_cmap() 

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# path = "/media/sc13967/Data1/DL/Nadia/2025-11-20_StarDist_Set1_400px_greyscale_norm/"
path = '/workspace/data/DL/Nadia/2025-11-21_StarDist_Set3_800px_RGB/'
X = sorted(glob(path+'train/images/*.tif'))
Y = sorted(glob(path+'train/masks/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

# axis_norm = (0,1)   # normalize channels independently
axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

conf = Config2D(n_channel_in=n_channel, train_batch_size=2, train_patch_size=(800,800), train_shape_completion=False)
print(conf)

model = StarDist2D(conf, name='stardist_no_shape_completion', basedir='models')

model_checkpoint = ModelCheckpoint('UNet_currentBest_E{epoch}_acc{acc:.3f}_ValLoss{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_best_only=False)
model.train(X_trn,Y_trn,validation_data=(X_val,Y_val))

model.export_TF()
