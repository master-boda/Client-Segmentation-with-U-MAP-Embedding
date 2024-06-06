import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def run_autoencoder(df, output_path, epochs=50, batch_size=32, latent_dim=6, random_state=42):
    if df.isnull().values.any():
        raise ValueError('Input DataFrame contains NaN values.')
    
    #define the random seed for reproducibility
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    # define autoencoder architecture
    input_dim = df.shape[1]

    # encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu', kernel_initializer='he_normal')(input_layer)
    encoded = Dense(32, activation='relu', kernel_initializer='he_normal')(encoded)
    latent_layer = Dense(latent_dim, activation='tanh', kernel_initializer='he_normal')(encoded)

    # decoder
    decoded = Dense(32, activation='relu', kernel_initializer='he_normal')(latent_layer)
    decoded = Dense(64, activation='relu', kernel_initializer='he_normal')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    # autoencoder model
    autoencoder = Model(input_layer, output_layer)

    # encoder model to extract the latent space
    encoder = Model(input_layer, latent_layer)

    # compile the autoencoder with Mean Squared Error loss function
    autoencoder.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True)

    # train the autoencoder
    autoencoder.fit(df, df, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

    # extract the latent representations
    latent_representation = encoder.predict(df)

    # convert latent representations to DataFrame
    latent_df = pd.DataFrame(latent_representation, index=df.index, columns=[f'latent_{i}' for i in range(latent_representation.shape[1])])

    latent_df.to_csv(output_path)

    return latent_df

#base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
#output_path = os.path.join(base_dir, 'latent_representation.csv')

#customer_info_path = os.path.join(base_dir, 'customer_info_preproc.csv')
#customer_info = pd.read_csv(customer_info_path, index_col='customer_id')

#latent_df = run_autoencoder(customer_info, output_path, epochs=50, batch_size=32, latent_dim=6)

#print(latent_df.head())
