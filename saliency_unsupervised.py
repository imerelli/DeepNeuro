####### In this script the saliency map is implemented for the unsupervised analysis. 
####### The saliency map, as we have implemented it, returns a rgb code for each gene.
####### The code represents the importance of the gene in the analysis.


import os

from vis.visualization import visualize_saliency
from vis.utils import utils
import numpy as np
import pandas as pd
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras_tqdm import TQDMCallback
from sklearn.model_selection import StratifiedShuffleSplit

conf = K.tf.ConfigProto(device_count={'CPU': 1},
                        intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4)
K.set_session(K.tf.Session(config=conf))


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
         'Whole_Blood': 'Whole_Blood.output_predicted_expression.txt'
         }


# Function for reparameterisation trick to make model differentiable
def sampling(args):
    import tensorflow as tf
    # Function with args required for Keras Lambda function
    z_mean, z_log_var = args

    # Draw epsilon of the same shape from a standard normal distribution
    epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                              stddev=epsilon_std)

    # The latent vector is non-deterministic and differentiable
    # in respect to z_mean and z_log_var
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z

class CustomVariationalLayer(Layer):
    """
    Define a custom layer that learns and performs the training
    This function is borrowed from:
    https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
    """

    def __init__(self, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_input, x_decoded):
        reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) -
                                K.exp(z_log_var_encoded), axis=-1)
        return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa

    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)


# Creating the big pandas dataframe

status0_data = pd.read_csv("ADNI0_cc_status.txt", header=None, sep=' ')
status0_data.set_index(0, inplace=True)
status0_data.rename(columns={1: 'status'}, inplace=True)
status0_data.index.names = ['id']

all_pandas = []
for f in sorted(files.items()):
    pd_tmp = pd.read_csv("data_adni_1/" + f[1], sep="\t").drop(columns=["FID"]).set_index("IID")
    pd_tmp = pd_tmp.join(status0_data)
    pd_tmp.rename(index=lambda x: x + "_" + f[0], inplace=True)
    all_pandas.append(pd_tmp)

all_df_before = pd.concat(all_pandas,sort=True)
print("DataFrame creation done")

all_df_ad = all_df_before[all_df_before['status'] == 1]
all_df_ad.drop(columns="status", inplace=True)
all_df_ctr = all_df_before[all_df_before['status'] == 0]
all_df_ctr.drop(columns="status", inplace=True)

# Running separately for the control and AD people
for (cur_status, all_df) in [('ad', all_df_ad), ('ctr', all_df_ctr)]:
    
    # Manually making the scaling because of NaNs
    all_df = all_df.sub(all_df.min()).div((all_df.max() - all_df.min()))
    print("DataFrame scalling done")
    all_df.fillna(0, inplace=True)
    print("DataFrame fillna done")

    # Uncomment to save a pickle with the dataframe for control and AD
    # all_df.to_pickle("all_df" + cur_status + ".pkl")

    # Split 20% test set randomly but keeping the ratio of each tissue
    # First, getting data's labels
    y_labels = []

    # Function used below to get the class labels in y_labels variable
    def label_race(row):
        global y_labels
        splitted = row.name.split('_')
        term = ""
        for i in range(3, len(splitted)):
            term += "_" + splitted[i]
        y_labels.append(term)
        return None

    # Run on our data
    all_df.apply(lambda row: label_race(row), axis=1)

    # Hyperparameters
    original_dim = all_df.shape[1]    
    epsilon_std = 1.0
    batch_size = 500
    beta = K.variable(0)
    epochs = 75              
    learning_rate = 0.001
    kappa = 1
    latent_dim = 42
    grads_ = []
    all_result = []


    # Creating and running 75 VAEs
    for step in range(75):      
        print("---------------------------------------------")
        print("\n\n")
        print("step",step)
        # Getting our data divided into stratified train/test parts
        # Be careful with StratifiedShuffleSplit class with n_splits > 2, might not be exactly what wanted
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)                         
        for train_index, test_index in sss.split(np.zeros(len(y_labels)), y_labels):
            pass

        gene_test_df = all_df.iloc[test_index].copy()
        gene_train_df = all_df.iloc[train_index].copy()

        # Encoder part as in tybalt's repository
        gene_input = Input(shape=(original_dim,), name="input")
        z_mean_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(gene_input)                
        z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
        z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

        z_log_var_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(gene_input)
        z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
        z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean_encoded, z_log_var_encoded])

        # Single decoding layer as in tybalt's repository
        decoder_to_reconstruct = Dense(original_dim, kernel_initializer='glorot_uniform', activation='sigmoid', name="output")
        gene_reconstruct = decoder_to_reconstruct(z)
        adam = optimizers.Adam(lr=learning_rate)
        vae_layer = CustomVariationalLayer()([gene_input, gene_reconstruct])
        vae = Model(gene_input, vae_layer)
        vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

        hist = vae.fit(np.array(gene_train_df),
                       shuffle=True,
                       epochs=epochs,
                       verbose=0,
                       batch_size=batch_size,
                       validation_data=(np.array(gene_test_df), None),
                       callbacks=[WarmUpCallback(beta, kappa),
                                  TQDMCallback(leave_inner=True, leave_outer=True)])

        ##### saliency-map

        layer_idx = utils.find_layer_idx(vae, 'input')
        seed = gene_train_df.values
       
        for m in range(original_dim):          
            for l in range(len(seed)):     

                grads = visualize_saliency(vae, layer_idx, filter_indices=[m], seed_input=seed[l])
                grads_.append(grads)
                result = ["status", cur_status, "step", str(step), "gene", str(m), "sample", str(l), "grads", str(grads)]
                result = '\t'.join(result)
                print(result)
                all_result.append(result)
             
        # Model to compress input
        encoder = Model(gene_input, z_mean_encoded)
        # Encode gene into the hidden/latent representation - and save output
        encoded_gene_df = encoder.predict_on_batch(all_df)
        encoded_gene_df = pd.DataFrame(encoded_gene_df, index=all_df.index)

        encoded_gene_df.columns.name = 'sample_id'
        encoded_gene_df.columns = encoded_gene_df.columns + 1
        encoded_file = os.path.join('multiple_vaes',
                                    "latent_" + cur_status + '_' + str(latent_dim) + '_' + str(step) + '_' + str(
                                        batch_size) + '_' + str(epochs) + '_' + str(learning_rate) + '_' + str(
                                        kappa) + '_encoded_gene_onehidden_warmup_batchnorm.tsv')
        # Saving the embedding space
        encoded_gene_df.to_csv(encoded_file, sep='\t')

        # Build a generator that can sample from the learned distribution
        decoder_input = Input(shape=(latent_dim,))
        _x_decoded_mean = decoder_to_reconstruct(decoder_input)
        decoder = Model(decoder_input, _x_decoded_mean)

    with open('vae_saliency_'+cur_status+'.txt', 'w') as f:
        for item in all_result:
            f.write("%s\n" % item)   

    with open('vae_grad_'+cur_status+'.txt', 'w') as f:
        for item in grads_:
            f.write("%s\n" % item)


