# Week 2 Progress Report

## Project Title

**Sentiment Classification of Tweets Using Deep Learning**

## 1. What Was Completed This Week

During Week 2, I worked on text preprocessing and baseline model implementation. The main goal of this week was to prepare the tweet text for machine learning and train the first simple model for comparison with the future deep learning model.

The Sentiment140 dataset was loaded again, and the original target labels were converted into binary sentiment classes. In the original dataset, label `0` represents negative sentiment and label `4` represents positive sentiment. For this project, they were converted into `0 = negative` and `1 = positive`.

I also created a text cleaning function to remove unnecessary noise from tweets. The cleaning process included lowercasing, removing links, replacing usernames, removing hashtag symbols, removing unnecessary characters, and deleting extra spaces.

After preprocessing, the dataset was divided into training, validation, and test sets. Then I trained the baseline model using TF-IDF features and Logistic Regression.

## 2. Important Commits or Files Changed

The main files prepared or updated during Week 2 are:

- `notebooks/week02_baseline.ipynb`
- `reports/week-02.md`
- `results/week02_baseline_results.csv`
- `README.md`

## 3. Experiments Run

This week, the main experiment was the baseline text classification model.

The following steps were completed:

- converted target labels into binary format;
- checked class distribution;
- cleaned tweet texts;
- removed empty texts after cleaning;
- split the dataset into train, validation, and test sets;
- transformed text into TF-IDF numerical features;
- trained Logistic Regression as the baseline model;
- evaluated the model on the validation set;
- evaluated the model on the test set;
- created a confusion matrix;
- inspected wrong predictions for basic error analysis;
- saved baseline results into the `results/` folder.

The baseline experiment used:

- Model: Logistic Regression
- Features: TF-IDF
- Task: Binary text classification
- Classes: negative and positive
- Metrics: accuracy, precision, recall, F1-score, confusion matrix

## 4. Results So Far

The baseline model was successfully implemented and evaluated.

The dataset was split into three parts:

| Split | Purpose |
|---|---|
| Training set | Used to train the model |
| Validation set | Used to check model performance during development |
| Test set | Used for final baseline evaluation |

The baseline model used TF-IDF features to convert tweets into numerical vectors. Logistic Regression was then trained to classify tweets as negative or positive.

The model performance was measured using:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix.

The results were saved in:

`results/week02_baseline_results.csv`

The baseline model gives the first comparison point for the project. This result will be compared with the LSTM deep learning model in Week 3.


## 5. Plan for Next Week

In Week 3, I plan to implement and train the deep learning model.

The planned tasks are:

- prepare tokenized text data for neural network training;
- build a vocabulary;
- convert tweets into sequences of integers;
- pad or truncate sequences to the same length;
- create PyTorch Dataset and DataLoader objects;
- implement an LSTM-based neural network;
- train the LSTM model;
- plot training and validation loss;
- evaluate the LSTM model using accuracy, precision, recall, F1-score, and confusion matrix;
- compare LSTM results with the Week 2 baseline model;
- write the Week 3 progress report.

## 6. Short Conclusion

Week 2 was completed successfully. The tweet text was cleaned, the dataset was split into training, validation, and test sets, and the first baseline model was trained using TF-IDF and Logistic Regression. The baseline results were saved for future comparison. The next step is to build a deep learning model using LSTM and compare it with the baseline approach.
