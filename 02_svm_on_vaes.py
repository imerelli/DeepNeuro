"""
Script used to check the latent representations of all the previously generated VAEs, and run the SVM procedure
as described in the paper.

This takes a lot of time to be concluded. For quickness is better to split the for cycles into different runs.
"""
import os
import pickle

import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
os.environ['KMP_DUPLICATE_LIB_OK']='True'

files = {'Adipose_Subcutaneous': 'Adipose_Subcutaneous.output_predicted_expression.txt',
         'Adipose_Visceral_Omentum': 'Adipose_Visceral_Omentum.output_predicted_expression.txt',
         'Adrenal_Gland': 'Adrenal_Gland.output_predicted_expression.txt',
         'Artery_Aorta': 'Artery_Aorta.output_predicted_expression.txt',
         'Artery_Coronary': 'Artery_Coronary.output_predicted_expression.txt',
         'Artery_Tibial': 'Artery_Tibial.output_predicted_expression.txt',
         'Brain_Amygdala': 'Brain_Amygdala.output_predicted_expression.txt',
         'Brain_Anterior_cingulate': 'Brain_Anterior_cingulate_cortex_BA24.output_predicted_expression.txt',
         'Brain_Caudate_basal_ganglia': 'Brain_Caudate_basal_ganglia.output_predicted_expression.txt',
         'Brain_Cerebellar': 'Brain_Cerebellar_Hemisphere.output_predicted_expression.txt',
         'Brain_Cerebellum': 'Brain_Cerebellum.output_predicted_expression.txt',
         'Brain_Cortex': 'Brain_Cortex.output_predicted_expression.txt',
         'Brain_Frontal_Cortex': 'Brain_Frontal_Cortex_BA9.output_predicted_expression.txt',
         'Brain_Hippocampus': 'Brain_Hippocampus.output_predicted_expression.txt',
         'Brain_Hypothalamus': 'Brain_Hypothalamus.output_predicted_expression.txt',
         'Brain_Nucleus': 'Brain_Nucleus_accumbens_basal_ganglia.output_predicted_expression.txt',
         'Brain_Putamen': 'Brain_Putamen_basal_ganglia.output_predicted_expression.txt',
         'Brain_Spinal_cord': 'Brain_Spinal_cord_cervical_c-1.output_predicted_expression.txt',
         'Brain_Substantia_nigra': 'Brain_Substantia_nigra.output_predicted_expression.txt',
         'Cells_EBV': 'Cells_EBV-transformed_lymphocytes.output_predicted_expression.txt',
         'Cells_Transformed': 'Cells_Transformed_fibroblasts.output_predicted_expression.txt',
         'Colon_Sigmoid': 'Colon_Sigmoid.output_predicted_expression.txt',
         'Colon_Transverse': 'Colon_Transverse.output_predicted_expression.txt',
         'Esophagus_Gastroesophageal': 'Esophagus_Gastroesophageal_Junction.output_predicted_expression.txt',
         'Esophagus_Mucosa': 'Esophagus_Mucosa.output_predicted_expression.txt',
         'Esophagus_Muscularis': 'Esophagus_Muscularis.output_predicted_expression.txt',
         'Heart_Atrial': 'Heart_Atrial_Appendage.output_predicted_expression.txt',
         'Heart_Left_Ventricle': 'Heart_Left_Ventricle.output_predicted_expression.txt',
         'Liver': 'Liver.output_predicted_expression.txt',
         'Lung': 'Lung.output_predicted_expression.txt',
         'Minor_Salivary': 'Minor_Salivary_Gland.output_predicted_expression.txt',
         'Muscle_Skeletal': 'Muscle_Skeletal.output_predicted_expression.txt',
         'Nerve_Tibial': 'Nerve_Tibial.output_predicted_expression.txt',
         'Pancreas': 'Pancreas.output_predicted_expression.txt',
         'Pituitary': 'Pituitary.output_predicted_expression.txt',
         'Skin_Not_Sun': 'Skin_Not_Sun_Exposed_Suprapubic.output_predicted_expression.txt',
         'Skin_Sun_Exposed': 'Skin_Sun_Exposed_Lower_leg.output_predicted_expression.txt',
         'Small_Intestine': 'Small_Intestine_Terminal_Ileum.output_predicted_expression.txt',
         'Spleen': 'Spleen.output_predicted_expression.txt',
         'Stomach': 'Stomach.output_predicted_expression.txt',
         'Thyroid': 'Thyroid.output_predicted_expression.txt',
         'Whole_Blood': 'Whole_Blood.output_predicted_expression.txt'}


# Adding tissue columns with tissue information
def label_race(row):
    splitted = row.name.split('_')
    term = ""
    for i in range(3, len(splitted)):
        term += "_" + splitted[i]
    return term


if __name__ == '__main__':
    # Hyperparameters first
    batch_size = 500
    epochs = 75
    learning_rate = 0.001
    kappa = 1
    latent_dim = 42

    for cur_status in ['ad', 'ctr']:
        for step in range(75):
            # Loading latent activations
            latent_dim_file = os.path.join('multiple_vaes',
                                           "latent_" + cur_status + '_' + str(latent_dim) + '_' + str(step) + '_' + str(
                                               batch_size) + '_' + str(epochs) + '_' + str(learning_rate) + '_' + str(
                                               kappa) + '_encoded_gene_onehidden_warmup_batchnorm.tsv')
            latent_dim_df = pd.read_csv(latent_dim_file, sep='\t')
            latent_dim_df.set_index('IID', inplace=True)

            latent_dim_w_tissue = latent_dim_df.copy()
            latent_dim_w_tissue['tissue'] = latent_dim_w_tissue.apply(lambda row: label_race(row), axis=1)

            step_dic = {}

            for i in range(latent_dim):
                print("########################################")
                print("########################################")
                print("### LATENT FEATURE " + str(i + 1) + " " + cur_status + str(step))
                X = latent_dim_w_tissue.iloc[:, i].values

                dic_latent_i = {}

                for f in sorted(files.items()):
                    print("--" + f[0])
                    y = latent_dim_w_tissue.loc[:, 'tissue'].copy().values
                    for j, elem in enumerate(y):
                        if elem == "_" + f[0]:
                            y[j] = 1
                        else:
                            y[j] = 0

                    dic_latent_i[f[0]] = {}

                    scoring = ['accuracy', 'f1', 'roc_auc']
                    reshaped_X = StandardScaler().fit_transform(X.reshape(-1, 1).copy())
                    list_y = list(y)
                    clf = svm.SVC(kernel='linear', class_weight="balanced", max_iter=100000)
                    scores = cross_validate(clf, reshaped_X, list_y, cv=5, scoring=scoring, n_jobs=-1)

                    score = scores['test_accuracy']
                    dic_latent_i[f[0]]['acc'] = score.mean()
                    dic_latent_i[f[0]]['acc_std'] = score.std()
                    print("Accuracy: %.4f (%.4f)" % (score.mean(), score.std()))

                    score = scores['test_f1']
                    dic_latent_i[f[0]]['f1'] = score.mean()
                    dic_latent_i[f[0]]['f1_std'] = score.std()
                    print("F1 score: %.4f (%.4f)" % (score.mean(), score.std()))

                    score = scores['test_roc_auc']
                    dic_latent_i[f[0]]['roc'] = score.mean()
                    dic_latent_i[f[0]]['roc_std'] = score.std()
                    print("ROC AUC: %.4f (%.4f)" % (score.mean(), score.std()))

                step_dic[i] = dic_latent_i
            # For each VAE, the results of the SVM are pickled for later use and analysis
            pickle.dump(step_dic, open(
                "multiple_vaes/latent_" + cur_status + '_' + str(latent_dim) + '_' + str(step) + "_svm_latent.pkl",
                "wb"))
