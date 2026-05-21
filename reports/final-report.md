# Final Report: Sentiment Classification of Tweets Using Deep Learning

## 1. Project Title

**Sentiment Classification of Tweets Using Deep Learning**

---

## 2. Problem Statement

### What problem are we solving?

Sentiment classification is the task of automatically identifying the emotional polarity of a text. In this project, the goal is to classify tweets as either **negative** or **positive** based on their text content.

The model receives a tweet as input and predicts one of two sentiment classes:

- `0` — Negative sentiment
- `1` — Positive sentiment

### Why is this useful?

Social media platforms contain a large amount of user opinions about products, services, brands, events, and public topics. Manually reading and analyzing thousands or millions of tweets is difficult and time-consuming.

An automatic sentiment classification system can help companies, researchers, and organizations understand public opinion faster. For example, it can be used to monitor customer feedback, analyze public reaction to events, or identify negative responses to a product or service.

### What does the model predict?

Given a tweet text, the model predicts a binary sentiment label:

| Label | Meaning |
|---|---|
| 0 | Negative |
| 1 | Positive |

This project belongs to the **text classification** task type.

---

## 3. Dataset Description

The dataset used in this project is the **Sentiment140 dataset with 1.6 million tweets** from Kaggle.

Dataset source:

https://www.kaggle.com/datasets/kazanova/sentiment140

The dataset file used in this project:

`training.1600000.processed.noemoticon`

In the local project folder, the dataset was stored as:

`д/training.1600000.processed.noemoticon`

### Dataset Information

| Field | Detail |
|---|---|
| Dataset name | Sentiment140 dataset with 1.6 million tweets |
| Source | Kaggle |
| Task type | Text classification |
| Full dataset size | 1,600,000 tweets |
| Input feature | Tweet text |
| Target label | Sentiment label |
| Classes | Negative, Positive |
| Data format | CSV |
| Dataset file | `training.1600000.processed.noemoticon` |

### Dataset Columns

The dataset does not include column names in the original file, so the following column names were assigned manually:

| Column | Description |
|---|---|
| target | Sentiment label of the tweet |
| ids | Tweet ID |
| date | Date of the tweet |
| flag | Query flag |
| user | Username |
| text | Tweet text |

### Label Mapping

In the original Sentiment140 dataset:

| Original Label | Meaning | New Label |
|---|---|---|
| 0 | Negative sentiment | 0 |
| 4 | Positive sentiment | 1 |

The original label `4` was converted to `1` to make the task a standard binary classification problem.

---

## 4. Data Preprocessing

### 4.1 Dataset Loading and Preparation

The raw Sentiment140 dataset was loaded from the local project folder. The dataset file name is:

`training.1600000.processed.noemoticon`

The original Sentiment140 file does not contain column names, so the following column names were assigned manually:

| Column | Description |
|---|---|
| target | Sentiment label |
| ids | Tweet ID |
| date | Date of the tweet |
| flag | Query flag |
| user | Username |
| text | Tweet text |

The dataset was loaded using `pandas` with `latin-1` encoding because the tweet text contains special characters.

The original sentiment labels were:

| Original Label | Meaning |
|---|---|
| 0 | Negative sentiment |
| 4 | Positive sentiment |

For this project, the labels were converted into binary format:

| Original Label | New Label | Meaning |
|---|---:|---|
| 0 | 0 | Negative |
| 4 | 1 | Positive |

The main input feature for the models is the `text` column.  
The target output is the converted binary sentiment label.

---

### 4.2 Text Cleaning

Tweets are usually noisy and informal. They may contain usernames, links, hashtags, abbreviations, spelling mistakes, emojis, and unnecessary symbols. Therefore, text cleaning was an important step before model training.

The cleaning process included:

- converting text to lowercase;
- converting HTML entities;
- removing URLs;
- replacing usernames with a common token;
- removing hashtag symbols while keeping the hashtag words;
- removing unnecessary symbols and numbers;
- removing extra spaces.

Example:

| Original Text | Cleaned Text |
|---|---|
| `@user I LOVE this movie!!! http://link.com` | `user i love this movie` |

The same cleaned text was used for the baseline model and the deep learning models.

---

### 4.3 Train / Validation / Test Split

The dataset was divided into three parts:

| Split | Percentage | Purpose |
|---|---:|---|
| Training set | 80% | Used to train the models |
| Validation set | 10% | Used to tune and monitor the models |
| Test set | 10% | Used for final evaluation |

The split was stratified. This means that the distribution of positive and negative classes was kept balanced in the training, validation, and test sets.

The test set was not used during training. It was used only at the end to evaluate how well the models perform on unseen tweets.

---

### 4.4 Feature Extraction

Different feature extraction methods were used for the baseline model and the deep learning models.

#### TF-IDF Features for Logistic Regression

For the baseline model, **TF-IDF Vectorization** was used. TF-IDF converts text into numerical vectors based on word importance.

The TF-IDF setup included:

| Parameter | Value |
|---|---|
| Features | Unigrams and bigrams |
| Maximum features | 50,000 |
| Minimum document frequency | 2 |
| Maximum document frequency | 0.95 |
| Model using these features | Logistic Regression |

TF-IDF was fitted only on the training set. The validation and test sets were transformed using the same vectorizer.

#### Word Sequences for LSTM and GRU

For the deep learning models, tweets were converted into sequences of word IDs.

The steps were:

- tokenize tweets into words;
- build vocabulary from the training set only;
- convert words into integer IDs;
- replace unknown words with `<UNK>`;
- pad shorter sequences with `<PAD>`;
- truncate longer sequences;
- save real sequence lengths.

The real sequence lengths were used with `pack_padded_sequence`, so LSTM and GRU could ignore padding tokens during training.

---

## 5. Model Architectures

This project compared three models:

| Model | Type |
|---|---|
| Logistic Regression + TF-IDF | Baseline model |
| LSTM | Deep learning model |
| GRU | Deep learning model |

---

### 5.1 Baseline Model — Logistic Regression + TF-IDF

The baseline model was **Logistic Regression with TF-IDF Vectorization**.

TF-IDF converted tweet text into numerical vectors. Logistic Regression then classified tweets as negative or positive.

This model was used as a baseline because it is:

- simple;
- fast;
- effective for text classification;
- easy to compare with deep learning models.

Model structure:

| Part | Description |
|---|---|
| Input | Cleaned tweet text |
| Feature extraction | TF-IDF Vectorization |
| Classifier | Logistic Regression |
| Output | Negative or Positive |

---

### 5.2 LSTM Model

The first deep learning model was **LSTM**, which stands for **Long Short-Term Memory**.

LSTM is suitable for text classification because tweets are sequences of words. Word order can change the meaning of a sentence. For example:

- `I like this`
- `I do not like this`

These two sentences have different meanings because of the word `not`.

The LSTM model included:

| Layer | Description |
|---|---|
| Embedding layer | Converts word IDs into dense vectors |
| Bidirectional LSTM layer | Learns sequence patterns from both directions |
| Dropout layer | Reduces overfitting |
| Fully connected layer | Produces final output |
| Output layer | Predicts negative or positive sentiment |

Real sequence lengths and `pack_padded_sequence` were used to help the model ignore padding tokens.

---

### 5.3 GRU Model

The second deep learning model was **GRU**, which stands for **Gated Recurrent Unit**.

GRU is similar to LSTM, but it has a simpler structure. It often trains faster because it has fewer gates than LSTM.

The GRU model included:

| Layer | Description |
|---|---|
| Embedding layer | Converts word IDs into dense vectors |
| Bidirectional GRU layer | Learns sequence patterns from both directions |
| Dropout layer | Reduces overfitting |
| Fully connected layer | Produces final output |
| Output layer | Predicts negative or positive sentiment |

GRU was added to compare two recurrent neural network models and analyze which one performs better for tweet sentiment classification.

---

## 6. Training Setup

All models were trained and evaluated using the same train, validation, and test splits.

### 6.1 Baseline Training Setup

| Parameter | Value |
|---|---|
| Model | Logistic Regression |
| Features | TF-IDF |
| Maximum features | 50,000 |
| N-grams | Unigrams and bigrams |
| Evaluation set | Test set |

The baseline model was trained using TF-IDF features from the cleaned tweet text. Logistic Regression was used because it is a strong and stable baseline for binary text classification.

---

### 6.2 Deep Learning Training Setup

| Parameter | LSTM / GRU |
|---|---|
| Environment | Google Colab |
| Framework | PyTorch |
| Embedding dimension | 128 |
| Hidden dimension | 128 |
| Batch size | 256 |
| Optimizer | AdamW |
| Loss function | BCEWithLogitsLoss |
| Epochs | 6 |
| Dropout | 0.4 |
| Sequence length | 50 |
| Padding handling | `pack_padded_sequence` |
| Best model selection | Validation F1-score |

The validation set was used to monitor model performance and select the best model based on F1-score.

The deep learning models were trained in Google Colab because PyTorch support and GPU runtime are more convenient there.

---

## 7. Evaluation Metrics

All models were evaluated on the test set using the same metrics.

### Accuracy

Accuracy shows the percentage of correctly classified tweets.

### Precision

Precision shows how many tweets predicted as positive were actually positive.

### Recall

Recall shows how many actual positive tweets were correctly found by the model.

### F1-score

F1-score balances precision and recall. It was used as the main metric for selecting the best model.

### Confusion Matrix

The confusion matrix shows correct and incorrect predictions for each class:

- True Negative;
- False Positive;
- False Negative;
- True Positive.

The best model was selected mainly based on **F1-score**, because F1-score provides a balanced view of precision and recall.

---

## 8. Results Table

The final comparison included three models:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression + TF-IDF | 0.8170 | 0.8386 | 0.7102 | 0.7691 |
| LSTM | 0.7902 | 0.7539 | 0.8616 | 0.8042 |
| GRU | 0.7931 | 0.7561 | 0.8635 | 0.8067 |

### Results Interpretation

The Logistic Regression + TF-IDF baseline achieved the highest accuracy and precision:

- Accuracy: `0.8170`
- Precision: `0.8386`

However, the deep learning models achieved higher recall and F1-score.

The GRU model achieved the highest F1-score:

- GRU F1-score: `0.8067`

The LSTM model also performed well:

- LSTM F1-score: `0.8042`

Based on F1-score, **GRU was selected as the best model** in this experiment.

Although Logistic Regression + TF-IDF was very strong, GRU gave the best balance between precision and recall.

---

## 9. Error Analysis

Error analysis was performed using predictions from the test set.

A prediction was considered correct when:

```text
true_label == predicted_label
```

A prediction was considered wrong when:

```text
true_label != predicted_label
```

The error analysis included:

- correct positive predictions;
- correct negative predictions;
- wrong GRU predictions;
- examples where LSTM and GRU predicted differently.

### Common Error Patterns

The models had difficulty with:

- very short tweets;
- informal language;
- slang and abbreviations;
- sarcasm and irony;
- tweets with unclear sentiment;
- noisy labels from the dataset.

Some tweets may contain positive words but express negative sentiment because of sarcasm. Other tweets may be too short to provide enough context for correct classification.

### LSTM and GRU Comparison

LSTM and GRU produced similar results because both are recurrent neural network models and were trained on the same processed dataset. However, GRU achieved a slightly higher F1-score, which means it performed slightly better overall in this experiment.

---

## 10. Limitations

This project has several limitations.

First, the task is binary classification. The model only predicts **negative** or **positive** sentiment. It does not include a neutral class.

Second, the Sentiment140 labels were created automatically. This means that some labels may be noisy or not fully accurate.

Third, tweets are short and informal. They often contain slang, abbreviations, emojis, hashtags, usernames, and links. This makes sentiment classification more difficult.

Fourth, LSTM and GRU were trained from scratch without pretrained word embeddings. Using pretrained embeddings or transformer-based models could improve performance.

Fifth, the models may not understand sarcasm, irony, or complex context very well.

Sixth, the project was trained and evaluated as a course project, so some experiments used samples for faster training. Training on the full dataset and tuning more hyperparameters could improve the results.

---

## 11. Conclusion

This project focused on sentiment classification of tweets using machine learning and deep learning. The main goal was to classify tweets as either **negative** or **positive**.

The project used the Sentiment140 dataset from Kaggle. The dataset contains tweet texts and sentiment labels.

Three models were implemented and compared:

1. Logistic Regression with TF-IDF Vectorization;
2. LSTM;
3. GRU.

The Logistic Regression + TF-IDF model was used as the baseline model. It achieved the highest accuracy and precision.

The LSTM and GRU models were implemented as deep learning models. They processed tweets as word sequences and used real sequence lengths to ignore padding tokens.

The final results showed that GRU achieved the highest F1-score:

```text
GRU F1-score = 0.8067
```

Therefore, **GRU was selected as the best model based on F1-score**.

The project shows that both traditional machine learning and deep learning can be effective for tweet sentiment classification. It also shows that deep learning models can achieve better recall and F1-score when sequence handling is implemented correctly.

Overall, the project completed the full machine learning pipeline: dataset exploration, preprocessing, baseline training, deep learning training, model comparison, error analysis, and final evaluation.

---

## 12. Future Improvements

In the future, the project can be improved by:

- using the full dataset instead of a sample;
- adding a neutral sentiment class;
- using pretrained word embeddings such as GloVe or Word2Vec;
- fine-tuning transformer models such as BERT or DistilBERT;
- improving emoji and hashtag processing;
- adding more detailed error analysis;
- creating a simple web demo for real-time tweet sentiment prediction;
- testing the model on real recent tweets from different topics;
- improving sarcasm and negation handling.

---


   https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
