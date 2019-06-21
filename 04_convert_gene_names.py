"""
Script used to convert the ensemble genes to gene names, to be included in the paper supplementary material.

It will print everything with latex commands as it was used in the paper latex source.
"""
import pandas as pd

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

ens_to_gene = pd.read_csv("ENSGid_to_gene_name.tsv", sep='\t').set_index("ENSGid")

for name, _ in sorted(files.items()):
    dict_arrs = dict()
    for reg in ['neg', 'pos']:
        for cond in ['ad', 'ctr']:
            df_tmp = pd.read_csv('multiple_vaes/results/' + name + '_' + reg + '_' + cond + '.txt', sep='\t',
                                 header=None)
            df_sorted = df_tmp.set_index(0).sort_values(by=1, ascending=False)
            df_sorted = df_sorted.head().join(ens_to_gene)
            dict_arrs[(reg, cond)] = [df_sorted.iloc[i, :].name if type(elem) != str else elem for i, elem in
                                      enumerate(df_sorted['gene_name'])]
    NO_LINES = 5
    for i in range(NO_LINES):
        if i == 0:
            print('\multirow{5}{*}{\\begin{minipage}{2.5cm} \\raggedright \\textbf{' + " ".join(
                name.split('_')) + '}\end{minipage}}', end='')
        else:
            print("\t\t\t\t\t\t\t\t", end='')
        line = []
        line.append(dict_arrs[('neg', 'ad')][i])
        line.append(dict_arrs[('neg', 'ctr')][i])
        line.append(dict_arrs[('pos', 'ad')][i])
        line.append(dict_arrs[('pos', 'ctr')][i])

        print(" & ", end='')
        print(" &\t".join(line), end=' \\\\')

        if i == 4:
            print(" \hdashline")
        else:
            print()
