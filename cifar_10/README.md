# [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

## 1. Setup Instructions
- Change the directory to cifar_10 using the following command `cd cifar_10`.
- The python is run in conda environment. To install the necessary packages run the following command: `conda env create -f environment_gpu.yml` (for gpu kernel) or `conda env create -f environment.yml` (for non gpu kernel).
- Use the following command to download the data : `bash download_data.sh`. This will create a directory `./cifar10`
- Use to following command to run python script the preprocess the image data: `python etl_image_data.py`. This will create a directory `./cifar10/numpy_data`.

## Directory Description
Some directories will be created as we run scripts.
- `./cifar10/` will house train and test folders which will contain train and test image data sets.
- `./cifar10/numpy_data` contains X and y data of train, valid, and test sets in .npy formats.
- `./models/` contains keras .h5 model files.
