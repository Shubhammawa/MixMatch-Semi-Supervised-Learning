# MixMatch
1. [Introduction](#Introduction)
2. [Colab Notebooks](#Colab-Notebooks)
3. [Dataset](#Dataset)
4. [Fully Supervised Baselines](#Fully-Supervised-Baselines)
5. [MixMatch utils](#MixMatch-utils)
6. [Training Logs](#Training-Logs)
7. [Usage](#Usage)

## Introduction
Problem Statement:
 - Gender Classification using face images.  
 - Mask vs Non-mask classifier using face images.  

Method: [MixMatch](https://arxiv.org/abs/1905.02249), a semi-supervised learning method published by [google-research](https://github.com/google-research).
## Colab Notebooks
1. MixMatch with and without mixup has been tested on the gender classification task. The colab notebooks for both are provided.
2. Extra Experimentation: Any other added functionality, tests etc are added in this folder.
    - A notebook with tensorboard visualization.
    - A notebook with visualizations of different data augmentations used.
    - A notebook for testing the mixup function and analysing hyperparameter effect on the distribution.

## Dataset
 - ### Gender Classification:
    All Ages Faces Dataset which comprises of around 13k images of people's faces and their corresponding age and gender information. We just need the gender label for this repository. The training dataset is divided into two parts: Labelled and Unlabelled. A separate test dataset is also created.
 - ### Mask vs Non-Mask Classification:
    AAF, Adience, MAFA and RWMF dataset are used for this task. Pre-processing is done in a similar manner as above.
## Fully Supervised Baselines
Fully supervised baselines with different number of labelled examples are trained.
## MixMatch Utils
 - Data augmentation, label guessing, sharpening and mixup functions.  
 - Dataset class for loading a batch of labelled and unlabelled dataset to pass to the MixMatch pipeline.
## Training Logs
Log files for loss functions, accuracy and f1-score over the epochs.

## Usage
MixMatch for image classification tasks can be run through the colab notebooks.
Two main variants, with and without mixup are provided.

 1. Imports:  
    All necessary libraries and utilities are imported here.

 2. Filepaths:  
    Paths for the csv files for training and testing data containing image names, labels and other metadata.
    Path for the zip folder containing all the images is also written here.  
    For most efficient use of google colab, images are uploaded as a zip file onto the drive and then the drive is mounted inside google colab.  
    Then the file is unzipped inside colab to access all the images.

 3. Hyperparameters:  
    All the relevant hyperparameters are defined here.
 
 4. Dataset Class:  
    An abstract class representing a dataset. The `__getitem__` and `__init__` methods are overwritten to read the filepaths for our dataset and images and labels are returned.

 5. Transformation:  
    A pytorch transform to be used in the DataLoader. The images are resized to a desirable size and then converted to tensors.

 6. Sample batch for visualization:  
    A small batch of the images is loaded and displayed along with their labels. Serves as a check to ensure proper loading of images and labels and other dimensionalty checks can also be done here.

 7. Datasets and Dataloaders:  
    The dataset objects instantiation is done here from the dataset class. Dataloaders are also created which help in loading a batch of images and their labels. Dataloaders have parameters for `batch_size`, `shuffle` etc.

 8. MixMatch utilities:  
    1. Data augmentation: Function which takes an image as an input and returns K number of augmentations of that image.
    2. Label guessing: Using the K augments of an images, the label is predicted after averaging over all the augmentations.
    3. Sharpening: The guessed labels are sharpened which implicitly minimizes the entropy of the output distribution, a desirable property and a standard practice in semi-supervised learning research.
    4. Mixup: Instead of passing images and their labels directly to the model, a linear combination of the images and their corresponding labels are passed to the model. This improves model robustness against adversarial examples. Link to the paper: [Mixup](https://arxiv.org/abs/1710.09412)

 9. MixMatch Dataset Class:
    A pytorch dataset class. The `__getitem__` and `__init__` methods are overwritten to take two different datasets as input and return a batch of lablled and unlabelled data.

 10. Loss Functions:  
    Labelled Loss: Cross Entropy  
    Unlabelled Loss: L2_Loss

 11. Wide-Resnet Architecture:  
    A pre-trained Wide-Resnet model is loaded from [torchvision models](https://pytorch.org/docs/stable/torchvision/models.html). This is the base CNN architecture on which a classifier as desired is stacked.

 12. Gender Classification Model:  
    A [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) class to define a classification model. Scores for each class are returned.

 13. Model Instantation:  
    The Wide-Resnet Architecture and Gender Classification Module are stacked using [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential).

 14. Optimizer:  
    Adam optimizer is used. The model parameters, learning rate and weight decay(regularization) are passed to the optimizer.

 15. Model training:  
     - A batch of labelled and unlabelled examples is loaded using the MixMatch Dataloader. 
     - Guessed labels are constructed for the unlabelled batch.
     - Images are passed to the model after applying mixup.
     - The loss functions are computed and added in the log file.
     - Gradients are backpropagated and model weights are updated.
     - F1 score and accuracy are computed on the test set.