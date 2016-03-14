import sklearn

from six.moves import cPickle as pickle
from sklearn import linear_model

a = pickle.load(open("notMNIST.pickle", "rb"))

(samples, labels) = sklearn.utils.shuffle(a['train_dataset'], a['train_labels'])

train_qty = 50
in_samples = samples[0:train_qty,:]
in_labels = labels[0:train_qty]

nsamples, nx, ny = in_samples.shape
in_samples_2d = in_samples.reshape((nsamples,nx*ny))
#in_samples_2d.shape

logreg = linear_model.LogisticRegression()

logreg.fit(in_samples_2d, in_labels)

(test_samples, test_labels) = sklearn.utils.shuffle(a['test_dataset'], a['test_labels'])

nsamples, nx, ny = test_samples.shape
test_samples_2d = test_samples.reshape((nsamples,nx*ny))
#test_samples_2d.shape

Z = logreg.predict(test_samples_2d[0:50])
Z
