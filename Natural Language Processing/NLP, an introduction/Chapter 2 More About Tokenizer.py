from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I, love my cat!',
    'You love my dog!',
    'What game do you play with dog?'
]

tokenizer = Tokenizer(num_words= 100, oov_token= '<OOV>')
# out of vocabulary words are fixed with it.
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

test_data = [
    'I really love my dog',
    'My dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)

print(word_index)
print(sequences)
print(test_seq)