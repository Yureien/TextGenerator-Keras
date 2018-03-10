from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import numpy as np
import random
import sys
import os
import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('weights',
                    help='''Path to the weights that were either pretrained or
                    generated with train.py.''')
parser.add_argument('-data', default='training_data.txt',
                    help='''Dataset to use for generating words. Should be same as one used for training.
                    Default: "training_data.txt"''')
parser.add_argument('-randomness', type=float, default=0.05,
                    help='''The exponential factor determining the predicted character
                    to be chosen. Do not change unless you know what you're doing. Default: 0.05''')
parser.add_argument('-length', type=int, default=500,
                    help='''Length of text to generate. Default: 500''')
parser.add_argument('-out_file', default='output.txt',
                    help='''Generated output. Default: "output.txt"''')
parser.add_argument('-seed', default='',
                    help='''Seed to use to generate the text.
                    Default: Chooses random text from the dataset.''')
args = vars(parser.parse_args())

if not os.path.isfile(args['weights']):
    print("Weights file not found!")

path = args['data']
with io.open(path, encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: 2 LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.load_weights(args['weights'])


start_index = random.randint(0, len(text) - maxlen - 1)
generated = ''
sentence = args['seed']
if sentence == '' or len(sentence) != 40:
    sentence = text[start_index: start_index + maxlen - 20]
generated += sentence
print('\n----- Generating with seed: "' + sentence.replace("\n", "\\n") + '" -----\n\n')
with open(args['out_file'], 'w') as f:
    sys.stdout.write(generated)
    f.write(generated)
    for i in range(args['length']):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = np.asarray(model.predict(x_pred, verbose=0)[0]).astype('float')
        preds = np.exp(np.log(preds*args['randomness']))
        preds /= np.sum(preds)
        preds = np.random.multinomial(1, preds, 1)
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        f.write(next_char)
        sys.stdout.flush()
        f.flush()
    f.write('\n')
    f.flush()
print()
print('----- DONE -----')
print("Written output to:", args['out_file'])
