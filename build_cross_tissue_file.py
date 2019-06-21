#### Construction of cross-tissue matrices

import pandas as pd
import numpy as np

file = ['Adipose_Subcutaneous.output_predicted_expression.txt',
         'Adipose_Visceral_Omentum.output_predicted_expression.txt',
         'Adrenal_Gland.output_predicted_expression.txt',
         'Artery_Aorta.output_predicted_expression.txt',
         'Artery_Coronary.output_predicted_expression.txt',
         'Artery_Tibial.output_predicted_expression.txt',
         'Brain_Amygdala.output_predicted_expression.txt',
         'Brain_Anterior_cingulate_cortex_BA24.output_predicted_expression.txt',
         'Brain_Caudate_basal_ganglia.output_predicted_expression.txt',
         'Brain_Cerebellar_Hemisphere.output_predicted_expression.txt',
         'Brain_Cerebellum.output_predicted_expression.txt',
         'Brain_Cortex.output_predicted_expression.txt',
         'Brain_Frontal_Cortex_BA9.output_predicted_expression.txt',
         'Brain_Hippocampus.output_predicted_expression.txt',
         'Brain_Hypothalamus.output_predicted_expression.txt',
         'Brain_Nucleus_accumbens_basal_ganglia.output_predicted_expression.txt',
         'Brain_Putamen_basal_ganglia.output_predicted_expression.txt',
         'Brain_Spinal_cord_cervical_c-1.output_predicted_expression.txt',
         'Brain_Substantia_nigra.output_predicted_expression.txt',
         'Cells_EBV-transformed_lymphocytes.output_predicted_expression.txt',
         'Cells_Transformed_fibroblasts.output_predicted_expression.txt',
         'Colon_Sigmoid.output_predicted_expression.txt',
         'Colon_Transverse.output_predicted_expression.txt',
         'Esophagus_Gastroesophageal_Junction.output_predicted_expression.txt',
         'Esophagus_Mucosa.output_predicted_expression.txt',
         'Esophagus_Muscularis.output_predicted_expression.txt',
         'Heart_Atrial_Appendage.output_predicted_expression.txt',
         'Heart_Left_Ventricle.output_predicted_expression.txt',
         'Liver.output_predicted_expression.txt',
         'Lung.output_predicted_expression.txt',
         'Minor_Salivary_Gland.output_predicted_expression.txt',
         'Muscle_Skeletal.output_predicted_expression.txt',
         'Nerve_Tibial.output_predicted_expression.txt',
         'Pancreas.output_predicted_expression.txt',
         'Pituitary.output_predicted_expression.txt',
         'Skin_Not_Sun_Exposed_Suprapubic.output_predicted_expression.txt',
         'Skin_Sun_Exposed_Lower_leg.output_predicted_expression.txt',
         'Small_Intestine_Terminal_Ileum.output_predicted_expression.txt',
         'Spleen.output_predicted_expression.txt',
         'Stomach.output_predicted_expression.txt',
         'Thyroid.output_predicted_expression.txt',
         'Whole_Blood.output_predicted_expression.txt'
         ]

def read_dataset():
    df = pd.read_csv("genes_cross_tissue.txt", header=None)     
    Y = df[df.columns[:]].values

    return Y

Y = read_dataset()

for data in range(len(file)):
    print(file[data])
    def read_dataset():
        df = pd.read_csv("data_adni_1/"+ file[data], delimiter="\t", header=None)
        X = df[df.columns[2:]].values

        return X

    X = read_dataset()

    gene = X[0]

    M = np.zeros(((len(X)-1), len(Y)))

    for j in range(len(X[0])):            
        for k in range(len(Y)):  
            for i in range(len(X)-1):   
                if Y[k][0] == gene[j]:
                    M[i][k] = X[i+1][j]

    np.savetxt("data_adni_1/cross_"+file[data], M, delimiter="\t")
