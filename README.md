# Malaria Cell Image Classifier
Malaria remains one of the most widespread and deadly infectious diseases, particularly across Africa and parts of Asia. Early and accurate diagnosis is critical for effective treatment and saving lives — yet manual detection under a microscope is time-consuming, subjective, and error-prone.  

This project uses **deep learning** to automate malaria detection from blood smear cell images, providing a fast, consistent, and low-cost diagnostic aid. It demonstrates how modern computer vision techniques can support medical professionals and expand diagnostic capacity in low-resource environments.

## Overview  
This project implements a model to classify individual blood-cell images as either **Parasitized** (i.e., infected with the malaria parasite) or **Uninfected**. It is based on a dataset of ~27,558 segmented cell images (equal numbers of parasitized/uninfected) from thin blood-smear slides. :contentReference[oaicite:0]{index=0}  

## Features  
- Splits the raw dataset into training, validation, and test sets.  
- Uses Keras’ ImageDataGenerator` for real‐time data augmentation (rotations, shifts, flips etc) to improve generalization.  
- Builds a convolutional neural network (CNN) based on a residual network (ResNet) variant, allowing deeper layers while preserving gradient flow.  
- Performs binary classification: an image → “Parasitized” or “Uninfected”.  
- At the end of training: outputs training/validation accuracy & loss plots, prints classification report (precision/recall/f1), and saves the trained model.  
- Structured so you can easily swap in other architectures or extend to multi‐class or other datasets.

## Getting Started  

### Prerequisites  
- Python 3.x  
- TensorFlow + Keras  
- OpenCV or similar image‐handling (for path collection)  
- NumPy, scikit‐learn, matplotlib   

### Dataset  
Download the cell-image dataset: ~27,558 images of thin blood smear cells, evenly balanced between parasitized and uninfected. :contentReference[oaicite:1]{index=1}  
Unzip into a folder structure such as:  

cell_images/
├── Parasitized/
└── Uninfected/

### Project Structure  
malaria-cell-image-classifier-keras/
├── cell_images/
│ ├── Parasitized/
│ └── Uninfected/
├── pyimagesearch/
│ ├── init.py
│ ├── config.py
│ └── resnet.py
├── build_dataset.py ← splits raw data into train/val/test sets
├── train_model.py ← trains the CNN model
├── evaluate_model.py ← evaluates on test set, prints report & saves model
├── plots/
│ └── training_plot.png ← sample output
└── README.md

### Usage  
1. Place the dataset into `cell_images/`.  
2. Run:  
   ```bash
   python build_dataset.py
to split into training, validation, and test folders.
3. Run: python train_model.py --plot plots/training_plot.png to train the CNN.
4. After training, run evaluate_model.py to see the classification report, confusion matrix, and to save the final model for future inference.

Configuration
In pyimagesearch/config.py you’ll find parameters you can adjust:

ORIG_INPUT_DATASET → path to the raw data.

TRAIN_SPLIT, VAL_SPLIT → ratios for splitting.

IMAGE_DIMS → size of images fed to the network (e.g., 64×64 or 224×224).

EPOCHS, BATCH_SIZE, etc.

Results
In the original tutorial, the model achieved roughly ~96% accuracy on this dataset. Your results may vary depending on hardware, augmentation, image size, architecture tweaks, etc.

Customisation / Next Steps
Try increasing the input image resolution (e.g., 128×128 or 224×224) and see if accuracy improves.

Swap in other architectures (DenseNet, Xception, MobileNet) or use transfer learning from ImageNet.

Convert the trained model into a mobile or web‐friendly format (TensorFlow Lite, ONNX) for deployment in low-resource settings.

Expand to multi-class classification (e.g., parasite life-cycle stages) or extend to other medical image datasets.
