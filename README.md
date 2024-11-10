# YouTube Comment Classification

This Python project implements machine learning classifiers to automatically classify YouTube comments from coding tutorial videos. The goal is to classify comments into two categories:

- **Content Concerns**: Comments related to the content of the video, such as questions, concerns, or suggestions for improvement.
- **Miscellaneous**: Comments that are irrelevant to the content, such as praise, spam, or insults.

The classifiers evaluated in this project include:

1. **Naïve Bayes Classifier**
2. **Support Vector Machine (SVM) Classifier**
3. **Decision Trees Classifier**
4. **Random Forest Classifier**

The classifiers are evaluated based on performance metrics such as accuracy, precision, recall, F1 score, and AUC. The results are compared across all classifiers, and 10-fold cross-validation is used for evaluation.

## Dataset

The dataset contains **6000 comments** (500 comments from 12 coding tutorial videos). These comments have been manually labeled by five annotators into two categories: **content concerns** and **miscellaneous**.

- **Content Concerns**: Questions, technical feedback, or suggestions related to the video.
- **Miscellaneous**: Non-technical comments like praise, insults, or irrelevant messages.

The dataset was originally collected from YouTube via the YouTube Data API.

## Features

- **Naïve Bayes Classifier**: A probabilistic classifier based on Bayes' Theorem.
- **Support Vector Machine (SVM)**: A powerful classifier that finds a hyperplane to separate classes.
- **Decision Trees**: A hierarchical model that splits data based on feature values.
- **Random Forests**: An ensemble method using multiple decision trees for improved accuracy.

The project includes a report comparing the results of each classifier, with a focus on evaluation metrics such as **accuracy**, **precision**, **recall**, and **F1 score**. Extra credit involves comparing the results to those reported in the original research paper and calculating the **AUC** and **ROC curve**.

## Requirements

- **Python 3.x**
- Libraries: `pandas`, `scikit-learn`, `matplotlib`, `numpy`, `seaborn`


