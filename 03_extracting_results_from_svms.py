"""
Script which uses the SVM results from the previous script in order to get a list of important genes, for each tissue,
both up- and down-regulated. Files will have, for each gene, a number. This number represents the amount of times that
gene appeared as up/down regulated for that tissue.
"""
import os
import pickle

import pandas as pd
from keras.models import load_model
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

latent_dim = 42


def save_to_file(cur_status, direction, tissue, the_list):
    with open('multiple_vaes/results/' + tissue + '_' + direction + '_' + cur_status + '.txt', 'w') as f:
        for item in the_list:
            f.write("%s\t%i\n" % item)

# Dictionaries where all the information will be stored (and then saved to disk)
all_dic = {}
all_dic['ctr'] = {}
all_dic['ctr']['pos'] = {}
all_dic['ctr']['neg'] = {}
all_dic['ad'] = {}
all_dic['ad']['pos'] = {}
all_dic['ad']['neg'] = {}

for cur_status in ['ctr', 'ad']:
    orig_columns = pd.read_pickle("all_df" + cur_status + ".pkl").columns
    for step in range(75):

        decoder_model_file = os.path.join('multiple_vaes', 'models_' + cur_status + '_' + str(latent_dim) + '_' + str(
            step) + '_500_75_0.001_1_decoder_onehidden_vae.hdf5')
        decoder = load_model(decoder_model_file)
        weights = []
        for layer in decoder.layers:
            weights.append(layer.get_weights())

        weight_layer_df = pd.DataFrame(weights[1][0], columns=orig_columns, index=range(1, 43))
        weight_layer_df.index.name = 'encodings'

        res = pickle.load(
            open("multiple_vaes/latent_" + cur_status + '_' + str(latent_dim) + '_' + str(step) + "_svm_latent.pkl",
                 "rb"))
        for i in range(latent_dim):
            for tissue_k, tissue_elem in res[i].items():
                if tissue_elem['f1'] > 0.80:
                    lat_weights = weight_layer_df.loc[[i + 1], :].T

                    # Checking tissue is in there
                    if tissue_k not in all_dic[cur_status]['pos']:
                        all_dic[cur_status]['pos'][tissue_k] = {}
                    if tissue_k not in all_dic[cur_status]['neg']:
                        all_dic[cur_status]['neg'][tissue_k] = {}

                    # Most positive
                    for gene in list(lat_weights.sort_values(by=i + 1, ascending=False).iloc[0:100, ].index):
                        if gene in all_dic[cur_status]['pos'][tissue_k]:
                            all_dic[cur_status]['pos'][tissue_k][gene] += 1
                        else:
                            all_dic[cur_status]['pos'][tissue_k][gene] = 1

                    # Most negative
                    for gene in list(lat_weights.sort_values(by=i + 1, ascending=True).iloc[0:100, ].index):
                        if gene in all_dic[cur_status]['neg'][tissue_k]:
                            all_dic[cur_status]['neg'][tissue_k][gene] += 1
                        else:
                            all_dic[cur_status]['neg'][tissue_k][gene] = 1

# Cycle to print information in the terminal, as well as save the files with the most important genes
for f in sorted(files.items()):
    print("For tissue " + f[0])
    # Positive
    pos_genes = [(x[0], x[1]) for x in all_dic['ctr']['pos'][f[0]].items() if x[1] > 3]
    print("\tCTR:\tGenes with most positive weights appearing more than 3 times: " + str(len(pos_genes)))
    save_to_file('ctr', 'pos', f[0], pos_genes)

    pos_genes_ad = [(x[0], x[1]) for x in all_dic['ad']['pos'][f[0]].items() if x[1] > 3]
    print("\tAD:\tGenes with most positive weights appearing more than 3 times: " + str(len(pos_genes_ad)))
    save_to_file('ad', 'pos', f[0], pos_genes_ad)

    # Negative
    neg_genes = [(x[0], x[1]) for x in all_dic['ctr']['neg'][f[0]].items() if x[1] > 3]
    print("\tCTR:\tGenes with most negative weights appearing more than 3 times: " + str(len(neg_genes)))
    save_to_file('ctr', 'neg', f[0], neg_genes)

    neg_genes_ad = [(x[0], x[1]) for x in all_dic['ad']['neg'][f[0]].items() if x[1] > 3]
    print("\tAD:\tGenes with most negative weights appearing more than 3 times: " + str(len(neg_genes_ad)))
    save_to_file('ad', 'neg', f[0], neg_genes_ad)

    print("\t-- Common positive genes between Control and AD: " + str(len(set(pos_genes) & set(pos_genes_ad))))
    print("\t-- Common negative genes between Control and AD: " + str(len(set(neg_genes) & set(neg_genes_ad))))
    print("\t-- Common genes among AD people (positive+negative): " + str(len(set(neg_genes_ad) & set(pos_genes_ad))))
    print("\t-- Common genes among Control people (positive+negative): " + str(len(set(neg_genes) & set(pos_genes))))
    print("\t-- Common genes among all sets: " + str(
        len(set(neg_genes) & set(neg_genes_ad) & set(pos_genes) & set(pos_genes_ad))))
