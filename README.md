# Project Proposal

## 1. Project Title

**Sentiment Classification of Tweets Using Deep Learning**

## 2. Problem Statement

The goal of this project is to build a text classification system that can automatically identify the sentiment of tweets. The model will analyze the text of a tweet and predict whether the sentiment is **positive** or **negative**.

This problem is useful because social media platforms contain a large amount of user opinions about products, services, events, brands, and public topics. Manually reading and analyzing thousands or millions of tweets is difficult and time-consuming. A sentiment classification model can help companies, researchers, and organizations understand public opinion faster.

In this project, the model will take tweet text as input and return one sentiment class as output:

- negative;
- positive.

This project belongs to the **text classification** task type.

## 3. Dataset

The dataset used in this project is the **Sentiment140 dataset with 1.6 million tweets** from Kaggle.

Dataset source link:

https://www.kaggle.com/datasets/kazanova/sentiment140

The dataset file used in this project is:

`training.1600000.processed.noemoticon`

In my project folder, the dataset is located in:

`д/training.1600000.processed.noemoticon`

Sentiment140 is a Twitter sentiment analysis dataset. It contains tweet texts with sentiment labels and is suitable for training and evaluating sentiment classification models.

The main input feature is the **tweet text**.  
The target output is the **sentiment label**.

The dataset contains the following columns:

| Column | Description |
|---|---|
| target | Sentiment label of the tweet |
| ids | Tweet ID |
| date | Date of the tweet |
| flag | Query flag |
| user | Username |
| text | Tweet text |

For this project, the main target labels are:

| Original Label | Meaning | New Label |
|---|---|---|
| 0 | Negative sentiment | 0 |
| 4 | Positive sentiment | 1 |

The project focuses on **binary sentiment classification**, where each tweet is classified as either negative or positive.

The data format is **CSV**. The dataset contains **1,600,000 tweets**, which makes it large enough for training, validation, and testing machine learning and deep learning models.

The dataset source will be clearly cited in the proposal, final report, and README file.

## 4. Planned Method

This project will include one baseline model and two deep learning models. The baseline model will be used as a comparison point, while the deep learning models will be used to test sequence-based neural network approaches.

### Data Preprocessing

Before training the models, the tweet texts will be cleaned and prepared.

The preprocessing steps include:

- loading the CSV file with the correct encoding;
- assigning column names;
- removing unnecessary columns if they are not needed;
- converting text to lowercase;
- removing links;
- removing or replacing usernames;
- removing extra spaces;
- handling hashtags;
- removing unnecessary symbols;
- tokenizing the text;
- converting labels into binary format;
- splitting the dataset into training, validation, and test sets.

### Baseline Model

The baseline model will be **Logistic Regression with TF-IDF Vectorization**.

TF-IDF will convert tweet texts into numerical vectors based on word importance. Logistic Regression will then classify the tweets as negative or positive.

This baseline model is useful because it is simple, fast, and effective for many text classification tasks. It also provides a comparison point before applying deep learning models.

### Deep Learning Models

The deep learning part of the project will include two recurrent neural network models:

1. **LSTM (Long Short-Term Memory)**
2. **GRU (Gated Recurrent Unit)**

LSTM and GRU were chosen because tweets are sequences of words, and word order can affect the meaning of a sentence. For example, the phrases **“I like this”** and **“I do not like this”** have different meanings because of the word **“not”**.

LSTM can learn longer context using memory cells. GRU is similar to LSTM but has a simpler structure and can train faster. By comparing both models, the project can analyze which recurrent neural network works better for tweet sentiment classification.

The deep learning models will include the following parts:

| Model Part | Description |
|---|---|
| Embedding layer | Converts words into dense numerical vectors |
| LSTM / GRU layer | Learns sequence patterns and context from tweet text |
| Dropout layer | Reduces overfitting |
| Fully connected layer | Connects learned features to final prediction |
| Output layer | Produces the final sentiment prediction |

The output layer will predict one of two classes:

- negative;
- positive.

### Loss Function

For LSTM and GRU models, the loss function will be **Binary Cross-Entropy with Logits Loss**.

This loss function is suitable because the project is a binary classification task with two classes: negative and positive.

### Evaluation Metrics

All models will be evaluated using the following metrics:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix;
- training and validation loss plots;
- validation F1-score plots.

These metrics will help compare the baseline model, LSTM model, and GRU model fairly.

### Train / Validation / Test Split

The dataset will be divided into:

| Split | Percentage |
|---|---|
| Training set | 80% |
| Validation set | 10% |
| Test set | 10% |

The training set will be used to train the models.

The validation set will be used to tune parameters and monitor overfitting.

The test set will be used only for final evaluation.

### Model Comparison

The final comparison will include three models:

| Model | Type |
|---|---|
| Logistic Regression + TF-IDF | Baseline model |
| LSTM | Deep learning model |
| GRU | Deep learning model |

The best model will be selected based on test set performance, especially **F1-score**, because F1-score balances precision and recall.

## 5. Expected Challenges

One expected challenge is that tweets are often noisy and informal. They may contain slang, abbreviations, hashtags, usernames, links, emojis, and spelling mistakes.

Another challenge is that the dataset labels are created automatically, so some labels may be noisy or not fully accurate. The models may also have difficulty understanding sarcasm, irony, negation, or very short tweets with little context.

Deep learning models may require more training time and memory than the baseline model. There is also a risk of overfitting if the neural network becomes too complex. To reduce this risk, dropout and validation monitoring will be used.

Another possible challenge is padding in sequence models. If LSTM or GRU reads padding tokens as part of the text, the model quality can decrease. To solve this, real sequence lengths and `pack_padded_sequence` will be used so that the models can ignore padding tokens.

## 6. Weekly Plan

| Week | Planned Work | Expected Output |
|---|---|---|
| Week 1 | Dataset selection, repository setup, dataset loading, exploratory data analysis | Proposal, README file, dataset summary |
| Week 2 | Text preprocessing, label encoding, train/validation/test split, baseline model training | Cleaned data, Logistic Regression + TF-IDF baseline results, Week 2 report |
| Week 3 | LSTM and GRU model implementation, training, evaluation, and comparison with baseline | Deep learning results, loss plots, F1-score plots, confusion matrices, Week 3 report |
| Week 4 | Final model comparison, error analysis, final report and presentation preparation | Final comparison table, final report, results files, slides/demo |

## Conclusion

This project focuses on building a tweet sentiment classification system using machine learning and deep learning. The models will classify tweets as either negative or positive.

The project includes one baseline model and two deep learning models:

1. Logistic Regression with TF-IDF Vectorization;
2. LSTM;
3. GRU.

The baseline model is used as a simple and effective comparison point. LSTM and GRU are used as deep learning models because they can process text as a sequence of words.

The final results will compare all three models using accuracy, precision, recall, F1-score, and confusion matrix. The best model will be selected based on test F1-score.

This project is useful because it shows how machine learning and deep learning can be applied to real social media text data for sentiment analysis.
