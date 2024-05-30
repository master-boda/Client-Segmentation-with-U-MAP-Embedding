import os
import pandas as pd
# Set environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define the file paths and read the dataset
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
customer_info_path = os.path.join(base_dir, 'customer_info_test.csv')
test = pd.read_csv(customer_info_path, index_col='customer_id')

# Define indices for continuous and binary variables
test_index = test.index
cont_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
binary_indices = [12, 13]
#test.reset_index(drop=True, inplace=True)
print(test.head())

binary_indices_tf = tf.constant(binary_indices)
continuous_indices_tf = tf.constant(cont_indices)

# Define the custom loss function
def custom_loss(y_true, y_pred):
    binary_loss = tf.keras.losses.BinaryCrossentropy()(tf.gather(y_true, binary_indices_tf, axis=1),
                                                       tf.gather(y_pred, binary_indices_tf, axis=1))
    mse_loss = tf.keras.losses.MeanSquaredError()(tf.gather(y_true, continuous_indices_tf, axis=1),
                                                  tf.gather(y_pred, continuous_indices_tf, axis=1))
    return tf.reduce_mean(binary_loss) + tf.reduce_mean(mse_loss)

# Define the autoencoder architecture
input_dim = test.shape[1]
latent_dim = 5

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
latent_layer = Dense(latent_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(32, activation='relu')(latent_layer)
decoded = Dense(64, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)  # Using sigmoid for output

# Autoencoder
autoencoder = Model(input_layer, output_layer)

# Encoder model to extract the latent space
encoder = Model(input_layer, latent_layer)

# Compile the autoencoder with the custom loss function
autoencoder.compile(optimizer='adam', loss=custom_loss)

# Train the autoencoder
autoencoder.fit(test, test, epochs=50, batch_size=32)

# Extract the latent representations
latent_representation = encoder.predict(test)

latent_df = pd.DataFrame(latent_representation, index=test_index, columns=[f'latent_{i}' for i in range(latent_representation.shape[1])])

print(latent_df.head())
latent_df.to_csv('latent_representation_test.csv', index=True)
