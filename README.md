# Deepfake Detection with Grad-CAM

This repository contains a Google Colab-ready notebook for detecting deepfakes using transfer learning with a pre-trained convolutional neural network (CNN) and visualizing model predictions with Grad-CAM.

## Overview

Deepfake technology has advanced rapidly, making it increasingly difficult to distinguish between real and manipulated media. This project aims to build a deepfake detection model capable of classifying videos as "real" or "fake" by analyzing frames extracted from them. Additionally, it incorporates Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations of the model's decisions, highlighting regions in the image that contribute most to its prediction.

The notebook performs the following steps:

1.  **Setup:** Installs necessary libraries and prepares the environment for Google Colab.
2.  **Dataset Download:** Downloads the `xdxd003/ff-c23` Kaggle dataset, which contains both real and deepfake videos.
3.  **Frame Extraction:** Extracts a fixed number of frames (default: 10) from each video, detects faces using MTCNN, crops them, and saves them into `data/real` and `data/fake` directories.
4.  **Data Preparation:** Creates `ImageDataGenerators` for training and validation, rescaling images and splitting the dataset.
5.  **Model Building:** Constructs a transfer learning model using a pre-trained EfficientNetB0 or MobileNetV2 backbone, with a custom classification head.
6.  **Training:** Trains the model for a lightweight demonstration (default: 2 epochs).
7.  **Evaluation:** Generates a classification report and confusion matrix on the validation set.
8.  **Grad-CAM Visualization:** Implements Grad-CAM to visualize activation heatmaps on sample images, showing which parts of the face the model focuses on.
9.  **Inference Helper:** Provides a simple function to predict whether an individual image is real or fake.

## Getting Started

### 1. Open in Google Colab

The easiest way to run this project is to open the provided notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/deepfake_detection_colab.ipynb)

*(Replace `YOUR_GITHUB_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub details if you fork this repo).*

### 2. Setup Kaggle API Key

To download the dataset, you'll need a Kaggle API key:

1.  Go to [Kaggle](https://www.kaggle.com/).
2.  Log in to your account.
3.  Click on your profile picture, then "My Account".
4.  Scroll down to the "API" section and click "Create New API Token". This will download `kaggle.json`.
5.  In your Colab environment, upload `kaggle.json` to the session storage or place it in `~/.kaggle/kaggle.json`. The notebook includes code to handle this if you upload it to the main directory.

### 3. Run the Notebook Cells

Execute each cell in the notebook sequentially.

*   The first cell (`0 - Runtime & Setup`) will install all necessary libraries.
*   The subsequent cells will download data, extract frames, train the model, and perform evaluations and visualizations.

## Code Structure

*   `deepfake_detection_colab.ipynb`: The main Colab notebook containing all the code and explanations.
*   `data/`: Directory where extracted frames are saved (created by the notebook).
    *   `data/real/`: Contains frames from real videos.
    *   `data/fake/`: Contains frames from deepfake videos.
*   `kaggle_dataset/`: Directory where the Kaggle dataset is downloaded and unzipped.

## Results & Visualizations

After training, the notebook will output a classification report and confusion matrix, demonstrating the model's performance.
