#%%
import tensorflow as tf

#%%
"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/sum(np.exp(x))

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

#%% Cross Entropy

cross_ent = sum(one_hot * log(softmax(scores)))

loss_fn = sum(cross_ent)/num_of_samples

#%%
y = 1000000000
for x in range(1000000):
    y+= 0.000001

y - 1000000000
#%%
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
#%%
import sklearn

a = pickle.load(open("notMNIST.pickle", "rb"))

(samples, labels) = sklearn.utils.shuffle(a['test_dataset'], a['test_labels'])

train_qty = 50
in_samples = samples[0:train_qty,:]
in_labels = labels[0:train_qty]

nsamples, nx, ny = in_samples.shape
in_samples_2d = in_samples.reshape((nsamples,nx*ny))
in_samples_2d.shape
#%%
from sklearn import linear_model
logreg = linear_model.LogisticRegression()

# we create an instance of Neighbours Classifier and fit the data.

logreg.fit(in_samples_2d, in_labels)
