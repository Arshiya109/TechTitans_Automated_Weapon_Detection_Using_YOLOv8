# Project Name

# TechTitans_Automated_Weapon_Detection_Using_YOLOv8

## Team Information

- **Grishma Yenchilwar** - [Team Leader]
- **Arshiya Saiyyad** - [Team member]
- **Ekta Kature** - [Team member]
- **Pranali Sharma** - [Team member] 
- **Vishakha Udapurkar** - [Team member]

## Mentor

- **Mr. Adhiroop Talasila** - [Mentor]

## Project Overview

[This project aims to perform custom object detection using the YOLOv8 model. We'll begin by annotating our dataset with the Label-studio tool. The annotated data will then be used to train the model on Google Colab. Finally, the trained model will be implemented in Visual Studio Code for detecting objects of interest, specifically knives, machine guns, pistols and non-weapon elements.]

# Automated Weapon Detection in Banks and ATMs

## Installation

### Prerequisites

- Google Drive account 
- Visual Studio Code
- Label-studio tool 
- Google colab 

### Step-by-Step Processing for the Setup

## Data Annotation via using the Label-studio

Label Studio is an open-source data labeling tool designed to handle a variety of annotation tasks. It provides a web-based interface for annotating images, text, and more, with support for collaborative workflows. After completing the annotations, we can export the data from the project dashboard via using the specific format that will suiti to model training requirements, such as YOLO-compatible formats or XML.
As the YOLOv8 formate is choosen as in export so it will give images, annoted images, class.json file and data.yaml file.

1. **Download and Install LabelImg**
   - Download Anaconda platform
   - Create the New Environment
   - Install Promt on Anaconda and download specific packages.

3. **Annotate Images**
   - Open Anaconda Prompt on system.
   - Give command as-
      ```sh
     conda activate [environment name]
     ```
     ```sh
     pip install label-studio
     ```
     ```sh
     label-studio
     ```
   - Go to Create.
   - Import the images for annotation.
   - Create bounding boxes around objects (knife, pistol, machine gun, non-weaponed elements).
   - Save the annotations in XML format ( TXT format (YOLO)).

   This annotated data will be used to train the YOLOv8 model.

4. **Export Annotations**
   - Ensure all annotations are correctly saved.
  

## Detailed Structure

**Training Data:**

- `dataset/train/images/`: This directory contains all the images used for training.
- `dataset/train/labels.txt`: This file includes the labels for each training image.

**Validation Data:**

- `dataset/val/images/`: This directory contains all the images used for validation.
- `dataset/val/labels.txt`: This file includes the labels for each validation image.

## Training YOLOv8 Model

We will use Google Colab to train the YOLOv8 model with the annotated data.

1. Prepare Data

- Uploading annotated images and annotation files to Google Drive.
- Providing Google Colab access to the Google Drive account.

2. Set Up Google Colab Environment

- Open a new notebook in Google Colab.
- Installing the `ultralytics` library:

4. Save the Trained Model

- Once the training process is complete, the model will generate weight files (`best.pt` and `last.pt`).
- These weight files will be located in the output directory specified during training.

## Implementing the Model in VS Code

To deploy the trained YOLOv8 model for object detection, it will follow these steps:

1. Set Up VS Code

- Installing the Python extension for Visual Studio Code.
- Setting up a new Python environment and activate it.

2. Load the Trained Model

- Copy the weight files (`best.pt` or `last.pt`) to your project directory.

3. Run the Detection Script

- Create a Python script to load and use the model:

  ```python
  from ultralytics import YOLO

  # Load the trained model
  model = YOLO('best.pt')

  # Perform detection on an image
  results = model.predict('path_to_image.jpg')

  # Display the results
  results.show()


4. **Display Results**
   - Ensuring the script runs without errors and displays detection results.

## Conclusion

This project demonstrates the process of creating a custom object detection model using YOLOv8. From annotating data with LabelImg, training the model on Google Colab, to implementing the model in VS Code, each step is crucial for achieving accurate and efficient object detection. With this guide, you can adapt the methodology to different datasets and objects.

