import os
import pandas as pd
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore

def run_autoencoder(df, output_path, epochs=50, batch_size=32, latent_dim=6):
    # define autoencoder architecture
    input_dim = df.shape[1]

    # encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    latent_layer = Dense(latent_dim, activation='relu')(encoded)

    # decoder
    decoded = Dense(32, activation='relu')(latent_layer)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)  # Using linear for continuous output

    # autoencoder model
    autoencoder = Model(input_layer, output_layer)

    # encoder model to extract the latent space
    encoder = Model(input_layer, latent_layer)

    # compile the autoencoder with Mean Squared Error loss function
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # train the autoencoder
    autoencoder.fit(df, df, epochs=epochs, batch_size=batch_size)

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
