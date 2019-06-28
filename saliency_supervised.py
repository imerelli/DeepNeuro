####### In this script the saliency map is implemented for the supervised analysis. 
####### We used a single-tissue matrix (Brain_Amygdala) as an example but it can be performed for all the matrices available.
####### The saliency map, as we have implemented it, gives filter/sample related information: each filter in the first layer
####### of the LSTM (150) analyzes all samples and for each of sample (in the cross-validation) the saliency function returns a rgb code.
####### The code represents the importance of the samples for the classification.     


from vis.visualization import visualize_saliency
from vis.utils import utils
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
import sklearn as sk
import keras
from keras import backend as K
from sklearn.model_selection import KFold
from keras import activations

import matplotlib.cm as cm

conf = K.tf.ConfigProto(device_count={'CPU': 1},
                        intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4)
K.set_session(K.tf.Session(config=conf))

callbacks = keras.callbacks.EarlyStopping(monitor='loss',
                              min_delta=0,
                              patience=2,
                              verbose=1,
                              mode='auto'
                              )

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def read_dataset():
    df = pd.read_csv("ADNI0_cc_status.txt", delimiter=" ", header=None)
    Y = df[df.columns[:]].values

    return Y

Y = read_dataset()

Y = Y[:600]

Y_status = []
Y_ID = []
for i in range(len(Y)):
    Y_status.append(Y[i][1])
    Y_ID.append(Y[i][0])

Y_status = np.asarray(Y_status)
Y_ID = np.asarray(Y_ID)

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

Y_status = dense_to_one_hot(Y_status, 2)
Y_status = Y_status.reshape(Y_status.shape[0], 1, Y_status.shape[1])

def read_dataset():
    df = pd.read_csv("/data_adni_1/Brain_Amygdala.output_predicted_expression.txt", delimiter="\t",header=0)  
    X = df[df.columns[2:]].values

    return X

X = read_dataset()
X = X[:600]

input_shape = X.shape[1]

####### Model Definition #######

N_features = input_shape
X = X.reshape(X.shape[0], 1, X.shape[1])

model = Sequential()
model.add(LSTM(150, input_shape = (None, N_features), return_sequences=True, name='input'))
model.add(BatchNormalization())
model.add(LSTM(10, return_sequences=True))
model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))

print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

kf = KFold(n_splits=10)
count = 0
mean_fpr = np.linspace(0, 1, 100)
grads_ = []
all_result=[]
layer_idx = utils.find_layer_idx(model, 'input')

i = 0
for train_index, test_index in kf.split(X):
    count = count + 1
    print("\nsplit...", count)
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = Y_status[train_index]
    y_test = Y_status[test_index]
    y_id_train = Y_ID[train_index]
    y_id_test = Y_ID[test_index]

    history = LossHistory()
    model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=0, callbacks=[history, callbacks])

    y_pred = model.predict(X_test)
    score = model.evaluate(X_test, y_test, verbose=0)
    y_classes = y_pred.argmax(axis=-1)

    #### saliency-map
    for m in range(150):  
        for l in range(len(X_train)):
                                
            train = X_train[l].reshape((X_train[l].shape[0],X_train[l].shape[1],X_train[l].shape[0]))
            
            grads = visualize_saliency(model, layer_idx, filter_indices=[m], seed_input=train)
            grads_.append(grads)
            result = ["split", str(count),"filter", str(m), "sample", str(l), str(y_id_train[l]), "grads", str(grads)]
            result = '\t'.join(result)
            print(result)
            all_result.append(result)
            
    i += 1

with open('Brain_Amygdala_saliency.txt', 'w') as f:
    for item in all_result:
        f.write("%s\n" % item)

with open('Brain_Amygdala_grad.txt', 'w') as f:
    for item in grads_:
        f.write("%s\n" % item)
