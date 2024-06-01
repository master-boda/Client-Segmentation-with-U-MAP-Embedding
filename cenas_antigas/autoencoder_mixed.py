import os
import pandas as pd
# disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
#from tensorflow.keras.callbacks import TensorBoard # type: ignore
#import datetime

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
customer_info_path = os.path.join(base_dir, 'customer_info_preproc.csv')
customer_info_preproc = pd.read_csv(customer_info_path, index_col='customer_id')

cont_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
binary_indices = [25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36]

binary_indices_tf = tf.constant(binary_indices)
continuous_indices_tf = tf.constant(cont_indices)

# custom loss function
def custom_loss(y_true, y_pred):
    binary_loss = tf.keras.losses.BinaryCrossentropy()(tf.gather(y_true, binary_indices_tf, axis=1),
                                                       tf.gather(y_pred, binary_indices_tf, axis=1))
    mse_loss = tf.keras.losses.MeanSquaredError()(tf.gather(y_true, continuous_indices_tf, axis=1),
                                                  tf.gather(y_pred, continuous_indices_tf, axis=1))
    return tf.reduce_mean(mse_loss) + tf.reduce_mean(binary_loss)

# autoencoder architecture
input_dim = customer_info_preproc.shape[1]
latent_dim = 4

# encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
latent_layer = Dense(latent_dim, activation='relu')(encoded)

# decoder
decoded = Dense(32, activation='relu')(latent_layer)
decoded = Dense(64, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)  # using sigmoid for output

# autoencoder
autoencoder = Model(input_layer, output_layer)

# encoder model to extract the latent space
encoder = Model(input_layer, latent_layer)

# compile the autoencoder with the custom loss function
autoencoder.compile(optimizer='adam', loss=custom_loss)

# train the autoencoder
autoencoder.fit(customer_info_preproc, customer_info_preproc, epochs=50, batch_size=32)

# extract the latent representations
latent_representation = encoder.predict(customer_info_preproc)

latent_df = pd.DataFrame(latent_representation, index=customer_info_preproc.index, columns=[f'latent_{i}' for i in range(latent_representation.shape[1])])

print(latent_df.head())
base_dir = 'data/processed'
latent_df.to_csv(os.path.join(base_dir, 'latent_representation.csv'))
