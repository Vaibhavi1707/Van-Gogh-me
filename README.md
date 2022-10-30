# Van-Gogh-me

This repository holds the source code for building and training a GAN to generate celeb portraits painted in, artist, Van Gogh's painting style. 

## Directory Structure

The repository has 3 sub directories, namely,

- `models`
    This directory, in turn, contains 2 sub-directories: 
    - `Objects` which contain the `.h5` files i.e. the files storing the model parameters. Such a file can be used to infer van gogh painting equivalents for new celebrity images.
    - `Scripts`: This directory contains scripts to build parts of the GAN such as the generator (`generator.py`) and discriminator (`discriminator.py`) networks in addition to their combination i.e. the final CycleGAN architecture (`CycleGAN.py`). This directory also contains a module (`losses.py`) collecting all loss functions used in training the CycleGAN model.

- `notebooks`
    This directory contains notebook which can be used to experiment with the constructed model.

- `preprocessing`
    This directory contains the script for collecting the painting images and celeb images in 2 different `tf.data.Datasets` objects. These objects when zipped together can be passed to the `model.fit` method, as done in `models/Scripts/train.py`.

The details on the archiecture and parameters are given in `Report.pdf`.