# README

## Overview
This project provides a pipeline for sub-event detection in Twitter streams. Two methodologies are implemented: One leveraging GloVe Twitter embeddings and various classification models, and a second one leveraging large language models (LLMs) and meta-modeling approaches. The following steps outline how to preprocess data, fine-tune an LLM, and calculate predictions using different methodologies. For more details on the context of this repository, the methodologies implemented, and the results achieved, please refer to `Data_challenge_report.pdf`.

## Prerequisites
Ensure that you have the following:
- Python 3.8 or higher
- The training and evaluation datasets placed in the respective folders:
  - `train_tweets` for the training dataset
  - `eval_tweets` for the evaluation dataset
- Required Python packages listed in `requirements.txt`

## Installation
Run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Instructions to calculate GloVe Twitter embeddings and train models to predict events
Run the `first_approach.ipynb` to load training datasets, calculate GloVe embeddings, perform feature engineering and train the model for the classification task.

## Instructions to fine-tune and run the LLM 

### Step 1: Preprocessing
Run the `preprocessing_tweets.ipynb` notebook to apply preprocessing to the training and evaluation datasets. This step prepares the data for further processing and ensures compatibility with the LLM.

### Step 2: Fine-Tuning the LLM
Execute the `llm_fine_tuning.py` script to fine-tune the LLM on the preprocessed training dataset. This step adjusts the LLM to perform well on the specific task of sub-event detection.

```bash
python llm_fine_tuning.py
```

### Step 3: Calculating Predictions Over Aggregates
Run the `calculate_predictions_over_aggregates.py` script to calculate tweet aggregates and obtain the model's predictions over these aggregates in both the training and evaluation datasets.

```bash
python calculate_predictions_over_aggregates.py
```

### Step 4: Calculating Predictions Per Period Using Thresholding
Run the `calculate_predictions_per_periods_threshold.py` script to compute the optimal threshold on the training dataset. This threshold is then used to generate predictions for the evaluation dataset based on aggregate-level predictions.

```bash
python calculate_predictions_per_periods_threshold.py
```

### Step 5: Using a Meta-Model for Period Predictions
Run the `calculate_predictions_over_periods_meta_model.ipynb` notebook to train an LSTM-based meta-model using the LLM's predictions over aggregates and additional features. The trained meta-model is then used to generate predictions for periods in the evaluation dataset.

### File Structure
The project folder should have the following structure:

```
project_folder/
├── train_tweets/                # Folder containing the training dataset
├── eval_tweets/                 # Folder containing the evaluation dataset
├── preprocessing_tweets.ipynb   # Notebook for preprocessing tweets
├── llm_fine_tuning.py           # Script for fine-tuning the LLM
├── calculate_predictions_over_aggregates.py  # Script for generating predictions over aggregates
├── calculate_predictions_per_periods_threshold.py  # Script for threshold-based period predictions
├── calculate_predictions_over_periods_meta_model.ipynb  # Notebook for meta-model predictions
├── requirements.txt             # List of required Python packages
```

## Notes
- Ensure that all datasets are properly preprocessed before running the fine-tuning or prediction scripts.
- Adjust hyperparameters in the scripts and notebooks as necessary to suit specific use cases or hardware constraints.

## Authors
Aziz Bacha and Wajdi Maatouk
