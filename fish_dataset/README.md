# LARGE SCALE FISH DATASET
This dataset contains 9 different seafood types collected from a supermarket in Izmir, Turkey for a university-industry collaboration project at Izmir University of Economics, and this work was published in ASYU 2020. The dataset includes gilt head bream, red sea bream, sea bass, red mullet, horse mackerel, black sea sprat, striped red mullet, trout, shrimp image samples.

## Purpose of the work
This dataset was collected in order to carry out segmentation, feature extraction, and classification tasks and compare the common segmentation, feature extraction, and classification algorithms (Semantic Segmentation, Convolutional Neural Networks, Bag of Features). All of the experiment results prove the usability of our dataset for purposes mentioned above.

## Data Gathering Equipment and Data Augmentation
Images were collected via 2 different cameras, Kodak Easyshare Z650 and Samsung ST60. Therefore, the resolution of the images are 2832 x 2128, 1024 x 768, respectively. Before the segmentation, feature extraction, and classification process, the dataset was resized to 590 x 445 by preserving the aspect ratio. After resizing the images, all labels in the dataset were augmented (by flipping and rotating). At the end of the augmentation process, the number of total images for each class became 2000; 1000 for the RGB fish images
and 1000 for their pair-wise ground truth labels.

## Description of the dataset
The dataset contains 9 different seafood types. For each class, there are 1000 augmented images and their pair-wise augmented ground truths. Each class can be found in the "Fish_Dataset" file with their ground truth labels. All images for each class are ordered from "00000.png" to "01000.png". For example, if you want to access the ground truth images of the shrimp in the dataset, the order should be followed is "Fish->Shrimp->Shrimp GT".