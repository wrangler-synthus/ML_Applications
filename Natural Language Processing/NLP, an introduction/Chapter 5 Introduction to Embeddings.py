# imdb datasets
import tensorflow as tf
import tensorflow_datasets as tfds
imdb, info = tfds.load('imdb_reviews', with_info= True, as_supervised= True)

import numpy as np

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(str(s.numpy().decode('utf8')))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy().decode('utf8')))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type  = 'post'
oov_tok = '<OOV>'

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = vocab_size , oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences ,
                       truncating = trunc_type,
                       maxlen=max_length)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,
                               maxlen= max_length)

reverse_word_index = dict([(val, key) for key, val in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Flatten(),
    # or equivalently
    # tf.keras.layers.GlobalAveragePooling1D
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(optimizer='adam', loss = 'binary_crossentropy',
              metrics= ['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))


model.save('models/imdb_embeddings')

e = model.layers[0]
weights = e.get_weights()[0]

import io

out_v = io.open('models/imdb_embeddings/vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('models/imdb_embeddings/meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + '\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')

out_v.close()
out_m.close()
#
# sentence = 'I really think this is amazing. honest'
# sequence = tokenizer.texts_to_sequences(sentence)
# print(sequence)
