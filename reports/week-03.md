# Week 3 Progress Report

## 1. What Was Completed This Week

During Week 3, I implemented and trained two deep learning models for tweet sentiment classification: **LSTM** and **GRU**.

The preprocessing, text cleaning, label conversion, and train/validation/test split were completed in Week 2. Therefore, in Week 3 I used the processed train, validation, and test files from Week 2 and focused only on deep learning model training.

I built a vocabulary from the training set only, converted tweets into sequences of word IDs, and used padding to make sequences the same length. To improve model quality, I also used real sequence lengths with `pack_padded_sequence`, so the models could ignore padding tokens during training.

The two deep learning models were:

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

Both models were trained, evaluated, and compared with the Week 2 baseline model.

## 2. Experiments Run

This week, I ran deep learning experiments for binary tweet sentiment classification.

The following steps were completed:

- uploaded Week 2 processed files to Google Colab;
- loaded train, validation, and test data;
- built vocabulary from the training set only;
- converted tweets into integer sequences;
- used real sequence lengths to ignore padding;
- created PyTorch Dataset and DataLoader objects;
- implemented an LSTM model;
- trained and evaluated the LSTM model;
- implemented a GRU model;
- trained and evaluated the GRU model;
- plotted training and validation loss;
- plotted validation F1-score;
- created confusion matrices;
- compared Logistic Regression + TF-IDF, LSTM, and GRU;
- saved model weights, vocabulary, results, and error analysis.

The models were evaluated using:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix.

## 3. Results So Far

The Week 3 comparison includes three models:

| Model | Type |
|---|---|
| Logistic Regression + TF-IDF | Baseline model |
| LSTM | Deep learning model |
| GRU | Deep learning model |

All models were evaluated on the same test set using accuracy, precision, recall, and F1-score.

### Model Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression + TF-IDF | 0.8170 | 0.8386 | 0.7102 | 0.7691 |
| LSTM | 0.7902 | 0.7539 | 0.8616 | 0.8042 |
| GRU | 0.7931 | 0.7561 | 0.8635 | 0.8067 |

Based on the results, **GRU achieved the highest F1-score among all models**. The GRU model reached an F1-score of **0.8067**, while LSTM reached **0.8042** and Logistic Regression + TF-IDF reached **0.7691**.

Although Logistic Regression + TF-IDF had the highest accuracy and precision, the deep learning models achieved higher recall and F1-score. Since F1-score balances precision and recall, **GRU was selected as the best model based on F1-score**.

LSTM and GRU produced similar results because both are recurrent neural network models and were trained on the same processed dataset. However, GRU showed slightly better performance in this experiment.

## 4. Plan for Next Week

In Week 4, I plan to complete the final project evaluation and final report.

The planned tasks are:

- load Week 3 results;
- compare all models in one final table;
- select the best model based on F1-score;
- complete final error analysis;
- prepare final report;
- prepare final presentation or demo;
- update README file;
- check repository structure;
- prepare for project defense questions.

## 5. Short Conclusion

Week 3 was completed successfully. I implemented and trained LSTM and GRU deep learning models for tweet sentiment classification. Both models were compared with the Week 2 baseline model using accuracy, precision, recall, F1-score, and confusion matrix.

The results showed that GRU achieved the highest F1-score, while Logistic Regression + TF-IDF achieved the highest accuracy and precision. Based on F1-score, GRU was selected as the best model in this experiment.

The next step is to complete final evaluation and prepare the final submission.
