import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
import sklearn as sk
from keras.models import load_model

from keras import backend as K

conf = K.tf.ConfigProto(device_count={'CPU': 1},
                        intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4)
K.set_session(K.tf.Session(config=conf))

# DATA
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

Y = Y[600:]     # test set
y_test_=Y

for k in range(len(matrix)):   
    model = load_model(matrix[k]+'.h5')
#    model = load_model(matrix[k]+'all_genes.h5')   
    def read_dataset():
        df = pd.read_csv("data_adni_1/"+matrix[k]+'.output_predicted_expression.txt', delimiter="\t", header=0)        # single tissue file
        X = df[df.columns[2:]].values
  
#        df = pd.read_csv("data_adni_1/cross_"+matrix[k]+'.output_predicted_expression.txt', delimiter="\t", header=None)        # cross-tissue file
#        X = df[df.columns[:]].values
        return X

    X = read_dataset()
    X = X[600:]    	# test set
    X = X.reshape(X.shape[0], 1, X.shape[1])

    predictions = model.predict(X)

    y_classes = predictions.argmax(axis=-1)
    fpr, tpr, thresholds = roc_curve(y_test_, y_classes)
    
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='(AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC - ADNI1 GWAS "+matrix[k])
    plt.legend(loc="lower right")
    plt.savefig(matrix[k]+"_test.png", bbox_inches='tight')
 #   plt.savefig(matrix[k]+"_test_all_genes.png", bbox_inches='tight')
    plt.close()

    count = 0
    for i in range(len(Y)):
        if y_classes[i]==Y[i]:
            count = count+1
    print("acc", '\t',float(float(count)*100/float(len(y_classes))),'\t',"auc",'\t',roc_auc, '\t', matrix[k])
