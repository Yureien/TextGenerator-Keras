# Keras Text Generator

This RNN feeds on a txt document (usually a book, recommended >500 KB) and then, after sufficient training (usually ~~a day (thanks, my laptop)~~ 10 epoches for humane sounding words and 50 epoches for actually meaningful sentences), it will spew out random sentences, based on a random seed.

The default training dataset, `training_data.txt`, is a mixture of several Harry Potter books.

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
python train.py
```
## Usage

### Simple version
Just run `python train.py`.
### Detailed version
Use `python train.py -h` for help. Output of that command -
```
usage: train.py [-h] [-data DATA] [-weights WEIGHTS] [-randomness RANDOMNESS]
                [-epochs EPOCHS] [-batch_size BATCH_SIZE] [-save_dir SAVE_DIR]

optional arguments:
  -h, --help            show this help message and exit

  -data DATA            Dataset to use for training. Recommended size >500KB.
                        Default: "training_data.txt"

  -weights WEIGHTS      If you want to resume from a trained weight, add the
                        path to the h5 weight here. The weights are
                        automatically saved each epoch.

  -randomness RANDOMNESS
                        The exponential factor determining the predicted
                        character to be chosen. Do not change unless you know
                        what you're doing. Default: 0.2

  -epochs EPOCHS        Number of epoches to do. I recommend >50 atleast.
                        Default: 200

  -batch_size BATCH_SIZE
                        Batch size. If you get a OutOfMemory error, reduce the
                        batch size. On big memory GPUs, you can increase this,
                        but not by much. Default: 128

  -save_dir SAVE_DIR    Directory where to save the weights. Default: weights
```