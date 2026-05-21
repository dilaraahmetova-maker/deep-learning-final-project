# Week 4 Progress Report

## Project Title

**Sentiment Classification of Tweets Using Deep Learning**

## 1. What Was Completed This Week

During Week 4, I completed the final evaluation of the tweet sentiment classification project.

The main goal of this week was to compare all implemented models and prepare the final outputs for the report and presentation. The final comparison included three models:

- Logistic Regression with TF-IDF Vectorization;
- LSTM;
- GRU.

I loaded the saved results from Week 3 and created the final model comparison table. The models were compared using accuracy, precision, recall, and F1-score. The best model was selected based on F1-score because this metric balances precision and recall.

I also loaded the saved LSTM and GRU models in Google Colab and tested them again on the test set. After that, I created confusion matrices and analyzed correct and wrong predictions from the test set.


## 2. Experiments Run

This week, I ran the final evaluation experiments.

The following steps were completed:

- loaded Week 3 model comparison results;
- compared Logistic Regression + TF-IDF, LSTM, and GRU;
- selected the best model based on F1-score;
- loaded the saved LSTM model;
- loaded the saved GRU model;
- created confusion matrices;
- analyzed wrong GRU predictions;
- compared cases where LSTM and GRU predicted differently;
- saved final result files for the final report and presentation.

## 3. Results So Far

The final comparison included three models:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression + TF-IDF | 0.8170 | 0.8386 | 0.7102 | 0.7691 |
| LSTM | 0.7902 | 0.7539 | 0.8616 | 0.8042 |
| GRU | 0.7931 | 0.7561 | 0.8635 | 0.8067 |

Based on the final results, **GRU achieved the highest F1-score**. Therefore, GRU was selected as the best model based on F1-score.

Logistic Regression + TF-IDF achieved the highest accuracy and precision, but GRU achieved better recall and F1-score. Since F1-score gives a balanced evaluation of precision and recall, GRU was selected as the final best model in this experiment.

The final comparison table was saved in:

`results/week04_final_model_comparison.csv`

Correct and wrong prediction examples were also saved for error analysis.

## 4. Short Conclusion

Week 4 was completed successfully. I compared Logistic Regression + TF-IDF, LSTM, and GRU. The final results showed that GRU achieved the highest F1-score and was selected as the best model based on F1-score.

The project now includes a complete pipeline: dataset exploration, preprocessing, baseline model, deep learning models, model comparison, error analysis, and final evaluation.
