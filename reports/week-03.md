# Week 3 Progress Report

## Project Title

**Sentiment Classification of Tweets Using Deep Learning**

## 1. What Was Completed This Week

During Week 3, I implemented and trained the main deep learning model for tweet sentiment classification. The main goal of this week was to move from the Week 2 baseline model to an LSTM-based neural network.

The model was trained in Google Colab because PyTorch was not available in the local notebook environment. I did not repeat the dataset loading, text cleaning, label conversion, or train/validation/test split from the previous weeks. Instead, I used the processed train, validation, and test files prepared in Week 2.

The text was tokenized, and a vocabulary was built from the training data only. Then the tweets were converted into sequences of integer IDs. Padding and truncation were used to make all sequences the same length for mini-batch training.

After that, I created PyTorch Dataset and DataLoader objects. Then I implemented an LSTM model with an embedding layer, LSTM layer, dropout layer, and fully connected output layer.

The LSTM model was trained and evaluated using accuracy, precision, recall, F1-score, and confusion matrix. The model result was compared with the Week 2 baseline model.

## 2. Important Commits or Files Changed

The main files prepared or updated during Week 3 are:

- `notebooks/03_deep_learning_model_colab.ipynb`
- `reports/week-03.md`
- `results/week03_lstm_results.csv`
- `results/week03_lstm_model.pt`
- `results/week03_vocab.json`

## 3. Experiments Run

This week, I ran the main deep learning experiment for binary tweet sentiment classification.

The following steps were completed:

- uploaded processed Week 2 files to Google Colab;
- loaded processed train, validation, and test files;
- tokenized tweet text;
- built vocabulary from the training set only;
- converted tweets into integer sequences;
- padded and truncated sequences to the same length;
- created PyTorch Dataset objects;
- created DataLoader objects;
- implemented an LSTM neural network;
- trained the LSTM model;
- plotted training and validation loss;
- plotted validation accuracy and F1-score;
- evaluated the LSTM model on the test set;
- created a confusion matrix;
- compared the LSTM model with the Week 2 baseline model;
- saved LSTM results, model weights, and vocabulary.

The LSTM model included:

- embedding layer;
- LSTM layer;
- dropout layer;
- fully connected output layer.

## 4. Results So Far

The LSTM deep learning model was successfully implemented and trained.

The Week 3 results were saved in:

`results/week03_lstm_results.csv`

The comparison includes:

| Model | Description |
|---|---|
| Logistic Regression + TF-IDF | Week 2 baseline model |
| LSTM | Main deep learning model |

The models were evaluated using:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix.

The LSTM model is suitable for this project because tweets are text sequences. Unlike TF-IDF with Logistic Regression, LSTM can process words in order and learn context from previous words. This is important because word order and negation can change the meaning of a sentence.


## 5. Plan for Next Week

In Week 4, I plan to finish the project and prepare the final submission.

The planned tasks are:

- compare baseline and LSTM results in one final table;
- analyze model mistakes in more detail;
- choose the best model;
- write the final report;
- prepare final presentation or demo;
- update README instructions;
- check repository structure;
- make sure all code and results are saved properly;
- prepare for project defense questions.

## 6. Short Conclusion

Week 3 was completed successfully. I trained and evaluated the LSTM deep learning model for tweet sentiment classification in Google Colab. The result was saved and compared with the Week 2 baseline model. The next step is to finalize the project, complete the final report, and prepare the presentation.
