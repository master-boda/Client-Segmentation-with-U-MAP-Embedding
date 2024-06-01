import os
import pandas as pd
# disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
customer_info_path = os.path.join(base_dir, 'customer_info_preproc_test.csv')
customer_info_preproc = pd.read_csv(customer_info_path, index_col='customer_id')

# autoencoder architecture
input_dim = customer_info_preproc.shape[1]
latent_dim = 6

# encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
latent_layer = Dense(latent_dim, activation='relu')(encoded)

# decoder
decoded = Dense(32, activation='relu')(latent_layer)
decoded = Dense(64, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)  # using linear for continuous output

# autoencoder
autoencoder = Model(input_layer, output_layer)

# encoder model to extract the latent space
encoder = Model(input_layer, latent_layer)

# compile the autoencoder with Mean Squared Error loss function
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# train the autoencoder
autoencoder.fit(customer_info_preproc, customer_info_preproc, epochs=50, batch_size=32)

# extract the latent representations
latent_representation = encoder.predict(customer_info_preproc)

latent_df = pd.DataFrame(latent_representation, index=customer_info_preproc.index, columns=[f'latent_{i}' for i in range(latent_representation.shape[1])])

print(latent_df.head())
latent_df.to_csv(os.path.join(base_dir, 'latent_representation.csv'))
