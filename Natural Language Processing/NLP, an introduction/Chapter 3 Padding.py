from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I, love my cat!',
    'You love my dog!',
    'What game do you play with dog?'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_indx = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding='post',
                       maxlen=5, truncating='post')
# padding <- change whether the padding before or after
# maxlen <- maximum length of the padded sentences
#  truncating <- where to lose the information from default from the beginning
# change it to end.
print(word_indx)
print(sequences)
print(padded)
