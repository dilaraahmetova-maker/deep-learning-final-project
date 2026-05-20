# Dataset Description

## Dataset Name

**Sentiment140 Dataset with 1.6 Million Tweets**

---

## Dataset Source

**Source:** Kaggle Dataset

**Exact dataset page link:**

```text
https://www.kaggle.com/datasets/kazanova/sentiment140
```

---

## Number of Examples

The dataset contains **1,600,000 tweets**.

This dataset is large enough for training and evaluating deep learning models. It is not a small toy dataset, so it is suitable for this project.

---

## Input Features

The main input feature used in this project is:

- **Tweet text**

The model will read the text of a tweet and learn whether the tweet expresses a positive or negative sentiment.

---

## Target Labels / Outputs

The original dataset uses sentiment labels:

| Original Label | Meaning |
|---|---|
| 0 | Negative sentiment |
| 4 | Positive sentiment |

For this project, the label **4** will be converted to **1**.

| Original Label | Project Label | Meaning |
|---|---|---|
| 0 | 0 | Negative tweet |
| 4 | 1 | Positive tweet |

Final model output:

| Label | Meaning |
|---|---|
| 0 | Negative |
| 1 | Positive |

---

## Data Format

The dataset is provided in **CSV format**.

---

## Main File Used

The main dataset file used in this project is:

```text
training.1600000.processed.noemoticon.csv
```

---

## Dataset Columns

The original CSV file does not include column names. During data loading, the following column names will be added:

| Column | Description |
|---|---|
| target | Sentiment label |
| id | Unique tweet ID |
| date | Date and time of the tweet |
| flag | Query flag |
| user | Twitter username |
| text | Tweet text |

---

# Download Instructions

## Option 1: Download from Kaggle Website

1. Open the official Kaggle dataset page:

```text
https://www.kaggle.com/datasets/kazanova/sentiment140
```

2. Click the **Download** button on Kaggle.

3. Download the dataset archive.

4. Extract the ZIP file.

5. Copy the following file into the `data/` folder:

```text
training.1600000.processed.noemoticon.csv
```

---

## Option 2: Download Using Kaggle API

From the project root folder, run:

```bash
kaggle datasets download -d kazanova/sentiment140 -p data/
```

Then unzip the downloaded file:

```bash
unzip data/sentiment140.zip -d data/
```

After extracting, check that the following file is inside the `data/` folder:

```text
training.1600000.processed.noemoticon.csv
```

---

## Option 3: Manual Copy After Extraction

If the dataset was downloaded and extracted in another folder, copy the main CSV file into this project folder:

```text
data/
```

The required file is:

```text
training.1600000.processed.noemoticon.csv
```

Expected location:

```text
data/training.1600000.processed.noemoticon.csv
```

---

# Expected Data Folder Structure

After preparing the dataset, the `data/` folder should look like this:

```text
data/
├── README.md
└── training.1600000.processed.noemoticon.csv
```

The ZIP file is not required after extraction.

Optional files that can be deleted after extraction:

```text
sentiment140.zip
```

---

# Important Note About GitHub

Large dataset files should **not** be uploaded to GitHub.

Do not upload these files:

```text
training.1600000.processed.noemoticon.csv
sentiment140.zip
```

These files should stay only on the local computer.

The repository should include download instructions instead of raw dataset files.

Recommended `.gitignore` rules:

```text
data/*.csv
data/*.zip
```

---

# License / Usage Notes

The dataset is publicly available on Kaggle.

It will be used only for educational and research purposes.

Large dataset files should not be uploaded to GitHub. The dataset should be stored locally in the `data/` folder, and the repository should include only download instructions.

---

# Why This Dataset Is Suitable

This dataset is suitable for the project because:

- it is a real-world Twitter dataset;
- it contains **1.6 million tweets**;
- it supports binary sentiment classification;
- it is large enough for training and evaluation;
- it can be used with both baseline and deep learning models.

---

# Project Task Connection

This dataset supports the project task:

**Text Classification / Sentiment Analysis**

The model will use:

```text
Input: tweet text
Output: sentiment label
```

The final goal is to classify each tweet as either **positive** or **negative**.
