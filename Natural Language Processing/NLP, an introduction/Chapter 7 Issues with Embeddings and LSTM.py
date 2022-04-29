# when sub-tokens are used
# the accuracy would be very low owing to the fact that
# sub-words do not often make sense, only words in a
#  sentence put together in a manner make sense.
# This info is used while making Recurrent Networks.

# Assume the same dataset, but only the models are changed

# Using LSTM
embedding_dims = None
vocab_size = None
max_length = None
import tensorflow as tf


# Single Bi-directional Model
model_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dims,
                              input_length = max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')

])

# Twin Bi-directional LSTM Model
# has smoother convergence than single Bi-directional LSTM
model_twin_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size ,embedding_dims,
                              input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences= True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation  = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# We can also invoke the use of Convolutions as done in Images
# We are using the values in 1D along features at a time

model_convolutions = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dims,
                              input_length= max_length),
    tf.keras.layers.Conv1D(128, 3, activation = 'relu'),
    tf.keras.layers.GlobalMaxPooling1D(2),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# so lets talk about training time and accuracy achieved

# The base model with only Embeddings <- ~ 171,533 parameters, 5 s per epoch
# Single LSTM <- 302,19 parameters, 43 s per epoch
# Single GRU <- 169,997 parameters, 20 s per epoch
# Convolution Networks <- 171,149 parameters, 6 s per epoch
