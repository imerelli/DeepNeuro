import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
import sklearn as sk
import keras
from keras import backend as K

conf = K.tf.ConfigProto(device_count={'CPU': 1},
                        intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4)
K.set_session(K.tf.Session(config=conf))

callbacks = keras.callbacks.EarlyStopping(monitor='loss',
                              min_delta=0,
                              patience=2,
                              verbose=1,
                              mode='auto')

# saving a list of losses over each batch during training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

### DATA
matrix = [
"Adipose_Subcutaneous",
"Adipose_Visceral_Omentum",
"Adrenal_Gland",
"Artery_Aorta",
"Artery_Coronary",
"Artery_Tibial",
"Brain_Amygdala",
"Brain_Anterior_cingulate_cortex_BA24",
"Brain_Caudate_basal_ganglia",
"Brain_Cerebellar_Hemisphere",
"Brain_Cerebellum",
"Brain_Cortex",
"Brain_Frontal_Cortex_BA9",
"Brain_Hippocampus",
"Brain_Hypothalamus",
"Brain_Nucleus_accumbens_basal_ganglia",
"Brain_Putamen_basal_ganglia",
"Brain_Spinal_cord_cervical_c-1",
"Brain_Substantia_nigra",
"Cells_EBV-transformed_lymphocytes",
"Cells_Transformed_fibroblasts",
"Colon_Sigmoid",
"Colon_Transverse",
"Esophagus_Gastroesophageal_Junction",
"Esophagus_Mucosa",
"Esophagus_Muscularis",
"Heart_Atrial_Appendage",
"Heart_Left_Ventricle",
"Liver",
"Lung",
"Minor_Salivary_Gland",
"Muscle_Skeletal",
"Nerve_Tibial",
"Pancreas",
"Pituitary",
"Skin_Not_Sun_Exposed_Suprapubic",
"Skin_Sun_Exposed_Lower_leg",
"Small_Intestine_Terminal_Ileum",
"Spleen",
"Stomach",
"Thyroid",
"Whole_Blood"
]

def read_dataset():
    df = pd.read_csv("ADNI0_cc_status.txt", delimiter=" ", header=None)       # case-control info file
    Y = df[df.columns[-1]].values

    return Y

Y = read_dataset()

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

Y = Y[:600]       # training set

Y_ = Y   # used to plot the roc curve
Y = dense_to_one_hot(Y, 2)
Y = Y.reshape(Y.shape[0], 1, Y.shape[1])

for j in range(len(matrix)):
    print(matrix[j])
    def read_dataset():
        df = pd.read_csv("data_adni_1/"+matrix[j]+".output_predicted_expression.txt", delimiter="\t", header=0)        # single tissue file 
        X = df[df.columns[2:]].values
       
 #       df = pd.read_csv("data_adni_1/cross_"+matrix[j]+".output_predicted_expression.txt", delimiter="\t", header=None)        # cross-tissue file       
 #       X = df[df.columns[:]].values
        return X

    X = read_dataset()
    X = X[:600]                 # training set
    
    input_shape = X.shape[1]
   
####### Model Definition #######

    N_features = input_shape
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()
    model.add(LSTM(150, input_shape = (None, N_features), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(10, return_sequences=True))
    model.add(BatchNormalization())    
    model.add(Dense(2, activation='softmax'))     

    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

###### Training ######

    kf = KFold(n_splits=10)
    count = 0
 
    cvscores = []
    tprs = []
    aucs = []
    precision = []
    recall = []
    f1 = []
    confusion = []

    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(X):
        count=count+1
        print("\nsplit...", count)
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = Y[train_index]
        y_test = Y[test_index]

        y_train_ = Y_[train_index]
        y_test_ = Y_[test_index]

        history = LossHistory()
        model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=0, callbacks=[history, callbacks])

        y_pred = model.predict(X_test)

        score = model.evaluate(X_test, y_test, verbose=0)
        print(score)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        cvscores.append(score[1] * 100)

        y_classes = y_pred.argmax(axis=-1)
        
        fpr, tpr, thresholds = roc_curve(y_test_, y_classes)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (count, roc_auc))
       	print("AUC", roc_auc)
        Precision = sk.metrics.precision_score(y_test_, y_classes)
        print("Precision", Precision)
        Recall = sk.metrics.recall_score(y_test_, y_classes)
        print("Recall", Recall)
        f1_score = sk.metrics.f1_score(y_test_, y_classes)
        print("f1_score", f1_score)
        confusion_matrix = sk.metrics.confusion_matrix(y_test_, y_classes)
        print("Confusion matrix","\n", confusion_matrix)

        aucs.append(roc_auc)
        precision.append(Precision)
        recall.append(Recall)
        f1.append(f1_score)
        confusion.append(confusion_matrix)

        i += 1

    model.save(matrix[j]+'.h5')
 #   model.save(matrix[j]+'_all_genes.h5')

    print("\n\n Scores:")
    print("mean acc ", "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("mean auc", np.mean(aucs))
    print("mean precision", np.mean(precision))
    print("mean recall", np.mean(recall))
    print("mean f1 score", np.mean(f1))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05]) 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC - ADNI1 GWAS "+matrix[j])
    plt.legend(loc="lower right")
    plt.savefig("ROC_"+matrix[j]+"_train.png", bbox_inches='tight')
 #   plt.savefig("ROC_"+matrix[j]+"_train_all_genes.png", bbox_inches='tight')
    plt.close()
    print("\n\n\n")
