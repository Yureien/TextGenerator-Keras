# Keras Text Generator

This RNN feeds on a txt document (usually a book, recommended >500 KB) and then, after sufficient training (usually ~~a day (thanks, my laptop)~~ 10 epoches for humane sounding words and 50 epoches for actually meaningful sentences), it will spew out random sentences, based on a random seed.

The default training dataset, `training_data.txt`, is a mixture of Harry Potter - Prisoner of Azkaban and Alice in Wonderland.

Sorry, I know this README is vague and sounds weird. I will update it later; busy with exams now.

#### Python3.6, Keras 2, Tensorflow 1.6 used.

## Installation

You need NumPY and Keras and Tensorflow installed. If you are a normal human who breaks under torture, I recommend using Tensorflow **only with GPU**.

#### I recommend not following below method. I installed via my OS'es package manager, as it had customized version of Tensorflow with GPU and CPU optimizations. (I belong to the Arch Linux masterrace). Install these your own way.
To install them, run (sorry, only for Linux (and maybe, Mac) users) -
```
pip install --user --upgrade numpy keras h5py tensorflow-gpu # Recommended, requires GPU.
```
**OR**
```
pip install --user --upgrade numpy keras h5py tensorflow # Not recommended, does not require GPU.
```

Then, as usual, git clone and cd to it's folder, and then, this -
```
python main.py
```
## Usage

### Simple version
Just run `python main.py`.
### Detailed version
Use `python main.py -h` for help. Output of that command -
```
usage: main.py [-h] [-data DATA] [-weights WEIGHTS] [-randomness RANDOMNESS]
               [-epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit

  -data DATA            Dataset to use for training. Recommended size >500KB.
                        Default is "training_data.txt".

  -weights WEIGHTS      If you want to resume from a trained weight, add the
                        path to the h5 weight here. The "weights-epoch-26.h5"
                        is attached as example.

  -randomness RANDOMNESS
                        Hard to explain. Initially, it should be around 0.2.
                        After around 30 epoches I change it to 0.4, and after
                        80, 0.8. As I said, hard to explain. Look at source
                        code. Default: 0.25

  -epochs EPOCHS        Number of epoches to do. I recommend >50 atleast.
                        Default is 200
```