# Week 1 Progress Report

## Project Title

**Sentiment Classification of Tweets Using Deep Learning**

## 1. What Was Completed This Week

During Week 1, I selected the topic and dataset for the final Applied Deep Learning project. The project topic is **Sentiment Classification of Tweets Using Deep Learning**. The task type is **text classification**, where the model will classify tweets as either **negative** or **positive**.

The selected dataset is the **Sentiment140 dataset with 1.6 million tweets** from Kaggle.

Dataset link:

https://www.kaggle.com/datasets/kazanova/sentiment140

This week, I also started working with the dataset in the first notebook. The dataset file used in the project is:

`training.1600000.processed.noemoticon`

The dataset is stored locally in the same folder as the notebook. I loaded the dataset, assigned column names, checked the dataset structure, inspected labels, and explored several tweet examples.

## 2. Important Commits or Files Changed

The main files prepared or updated during Week 1 are:

- `README.md`
- `data/README.md`
- `notebooks/01_dataset_exploration.ipynb`
- `reports/week-01.md`
- `requirements.txt`

Recommended GitHub commits for Week 1:

    git add README.md data/README.md requirements.txt
    git commit -m "Set up project structure and documentation"

    git add notebooks/01_dataset_exploration.ipynb
    git commit -m "Add dataset loading and initial exploration notebook"

    git add reports/week-01.md
    git commit -m "Add Week 1 progress report"

## 3. Experiments Run

This week focused on dataset understanding and exploratory data analysis. No machine learning model was trained yet.

The following experiments and checks were completed:

- loaded the Sentiment140 dataset file;
- assigned the column names: `target`, `ids`, `date`, `flag`, `user`, `text`;
- checked the number of rows and columns;
- checked data types;
- checked missing values;
- inspected original target labels;
- converted the target labels for binary classification;
- viewed examples of negative and positive tweets;
- calculated tweet character length;
- calculated tweet word count;
- created a simple text cleaning preview.

## 4. Results So Far

The dataset was successfully loaded after fixing the file path and CSV reading issue.

The Sentiment140 dataset contains tweet text and sentiment labels. The original labels are:

| Original Label | Meaning |
|---|---|
| 0 | Negative |
| 4 | Positive |

For this project, the labels are converted into binary format:

| New Label | Meaning |
|---|---|
| 0 | Negative |
| 1 | Positive |

The main input feature for the model will be the `text` column. The target variable will be the converted sentiment label.

Initial exploration showed that tweets contain noisy and informal text. Many tweets include usernames, links, hashtags, abbreviations, spelling mistakes, and emotional expressions. This means that text preprocessing will be important before training the models.

## 5. Problems or Blockers

The first problem was that the dataset file was not found because the notebook and dataset were stored in the same folder. The path code was corrected so that the notebook searches for the dataset in the current folder.

Another issue was a CSV reading error:

`EOF inside string`

This happened because one row in the dataset had a parsing problem. The loading code was updated to use the Python CSV engine and skip bad lines.

A possible future blocker is the large dataset size. Since the dataset contains 1.6 million tweets, training may take time and require enough memory.

## 6. Plan for Next Week

In Week 2, I plan to focus on text preprocessing and the baseline model.

The planned tasks are:

- clean tweet text;
- remove or replace links and usernames;
- handle hashtags and extra spaces;
- convert text into numerical features using TF-IDF;
- split the data into training, validation, and test sets;
- train a baseline Logistic Regression model;
- evaluate the baseline using accuracy, precision, recall, F1-score, and confusion matrix;
- save baseline results;
- write the Week 2 progress report.

## 7. Short Conclusion

Week 1 was completed successfully. The project topic and dataset were selected, the first notebook was created, and the dataset was loaded and explored. The main dataset issues were fixed. The next step is to preprocess the tweet text and build the first baseline model.
