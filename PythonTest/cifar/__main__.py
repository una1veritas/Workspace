import sys
import pickle
import numpy as np

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data

X_train = None
y_train = []

for i in range(1,6):
    data_dic = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
    if i == 1:
        X_train = data_dic['data']
    else:
        X_train = np.vstack((X_train, data_dic['data']))
    y_train += data_dic['labels']

test_data_dic = unpickle("cifar-10-batches-py/test_batch")
X_test = test_data_dic['data']
X_test = X_test.reshape(len(X_test),3,32,32)
y_test = np.array(test_data_dic['labels'])
X_train = X_train.reshape((len(X_train),3, 32, 32))
y_train = np.array(y_train)
