from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import numpy as np
import random
import sys
import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data', default='training_data.txt',
                    help='''Dataset to use for training. Recommended size >500KB.
                    Default is "training_data.txt".''')
parser.add_argument('-weights', default='',
                    help='''If you want to resume from a trained weight, add the path to
                    the h5 weight here. The "weights-epoch-26.h5" is attached as example.''')
parser.add_argument('-randomness', type=float, default=0.25,
                    help='''Hard to explain. Initially, it should be around 0.2.
                    After around 30 epoches I change it to 0.4, and after 80, 0.8.
                    As I said, hard to explain. Look at source code. Default: 0.25''')
parser.add_argument('-epochs', type=int, default=200,
                    help='''Number of epoches to do. I recommend >50 atleast.
                    Default is 200''')
args = vars(parser.parse_args())

path = args['data']
with io.open(path, encoding='utf-8') as f:
    text = f.read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
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


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text. Also saves model.
    if epoch % 50 == 0 and epoch != 0:
        model.save("trained_model_complete_%d.h5" % (epoch,))

    model.save("trained_model_weights_%d.h5" % (epoch,))
    print("Saved model.")

    print()
    print('----- Generating text after Epoch: %d -----' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence.replace("\n", "\\n") + '" -----')
    print()
    sys.stdout.write(generated)

    for i in range(400):
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
        sys.stdout.flush()
    print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
if args['weights'] != "":
    model.load_weights(args['weights'])
model.fit(x, y,
          batch_size=128,
          epochs=args['epochs'],
          callbacks=[print_callback])
