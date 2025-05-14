# Voice Gender Recognition

This repository contains my personal reimplementation of the paper named ["Voice Gender Recognition Using Deep Learning"](https://www.atlantis-press.com/proceedings/msota-16/25868884) proposed by Mucahit Buyukyilmaz and Ali Osman Cibikdiken. All credits goes to the authors.


## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Directory Structure](#directory-structure)
- [Training the Model](#training-the-model)
- [Using the Model](#using-the-model)
- [Logic Behind the Model](#logic-behind-the-model)
- [License](#license)

## Installation

To install this package, firstly clone the repository to the directory of your choice
Finally, you need to create a conda environment and install the requirements. This can be done using conda or pip. For `conda` use the following command:
```bash
conda create --name vgr --file requirements.txt python=3.10
conda activate vgr
```


## Dataset

The model requires a specific dataset to function correctly. Download the dataset from this Kaggle link. Ensure the dataset is organized as per the expected structure, as the code may not work with other datasets or folder arrangements. [this link](https://www.kaggle.com/datasets/primaryobjects/voicegender).

## Directory Structure

```bash
├── __init__.py
├── LICENSE
├── notebook/
│   └── visualize_training.ipynb
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
└── train.py
```

Explaining briefly the main folders and files:

* `train.py`: The main script for training the model.
* `src`: Contains core modules for dataset handling, model creation, and utility functions.
* `requirements.txt`:Lists the dependencies required for the project.
* `notebook`: Contains Jupyter notebooks for visualizing training, validation, and testing results.


## Training the Model

To train the model, run the following command:
```python3
python train.py -i CSV_FILE_PATH -o OUTPUT_DIR -l LOGGING_DIR
```

### Inputs Parameters

The following arguments **MUST** be passed:

* `-i` or `--input_dir`: Path to the input CSV file containing the dataset.
* `-o` or `--output_dir`: Directory where the best model checkpoints will be saved.
* `-l` or `--logging_dir`: Directory where training logs will be stored.
 
## Using the Model

Once the model is trained, it can be used for inference. To use the trained model:

### Load the saved model checkpoint from the output directory.
### Preprocess the input data (e.g., extract features from voice samples).
### Pass the preprocessed data to the model for prediction.

You can extend the repository by creating an inference script or notebook to automate these steps.

## Logic Behind the Model

The model leverages deep learning techniques to classify gender based on voice features. The process involves the following steps:


 
### Feature Extraction:
Acoustic features such as pitch, frequency, and amplitude are extracted from the audio data. These features are stored in a CSV file.
### Data Preprocessing:
The extracted features are normalized and prepared for input into the model.
### Model Architecture:
A deep learning model (e.g., a feedforward neural network) is trained on the preprocessed features. The model learns patterns and relationships between the features and the target labels (male or female).
### Training: 
The model is trained using a supervised learning approach, optimizing a loss function to minimize classification errors.
### Evaluation: 
The trained model is evaluated on a validation/test set to measure its accuracy and generalization performance.
### Usage:
During training we save a model on each fold, we found that best results are when we use all folds of the model using a voting structure


## License
This project is licensed under the MIT License. See the LICENSE file for details.