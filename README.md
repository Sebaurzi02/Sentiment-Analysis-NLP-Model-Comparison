# Sentiment Analysis and Text Classification with Machine Learning and Natural Language Processing (NLP)


This repository contains **two Jupyter Notebooks** that demonstrate a complete **Sentiment Analysis** pipeline using three  machine learning algorithms: **Logistic Regression (LR)**, **Multinomial Naive Bayes (MNB)**, and **Linear Discriminant Analysis (LDA)**. The objective is to classify text data into **positive** or **negative** sentiment classes, compare the performance of the models, and draw meaningful conclusions based on the results.

A core part of the pipeline focuses on **Natural Language Processing (NLP)** techniques for **text cleaning and optimization**, which include:

* Removing duplicates and noisy elements
* Stripping HTML tags
* Converting text to lowercase
* Removing punctuation and **stop words**
* **Tokenization** and **stemming**

These preprocessing steps are essential for improving the quality and consistency of the text data, enabling more effective feature extraction using both **Bag of Words (BoW)** and **TF-IDF (Term Frequency-Inverse Document Frequency)** representations.

The notebooks provide detailed comparisons of the three models trained with both vectorization techniques, along with metrics such as accuracy, precision, recall, F1-score, confusion matrices, and runtime. The analysis concludes by identifying the most effective method for this binary sentiment classification task.

---

 # Dataset: IMDb
The IMDb dataset consists of 50,000 rows of data containing two main columns, namely "review," which contains the text of English movie reviews, and "sentiment," which indicates the sentiment label as positive (1) or negative (0). Here are the detailed explanations

### Table: Structure of the IMDb Dataset

| No. | Column    | Data Type     | Description                                                             |
|-----|-----------|---------------|-------------------------------------------------------------------------|
| 1.  | review    | String        | Text of English movie reviews                                           |
| 2.  | sentiment | Integer       | Target column indicating the sentiment label as positive (1) or negative (0) based on the review |

---

## 1.0 Notebook Overview

The main notebook (`Sentiment_Analysis.ipynb`) walks through the complete process of preparing the data and training classification models using two different vectorization techniques: **Bag of Words (BoW)** and **TF-IDF (Term Frequency-Inverse Document Frequency)**.

###  Steps :

1. ### **Text Preprocessing**

   * Duplicate removal
   * Elimination of noise (special characters, punctuation, etc.)
   * HTML tag stripping
   * Lowercasing, stemming, and tokenization using basic NLP techniques

2. ### **Model Training**

   * Vectorization of cleaned text using both **BoW** and **TF-IDF**
   * Training of two classifiers:

     * Logistic Regression (LR)
     * Multinomial Naive Bayes (MNB)

3. ###  **Model Evaluation**

   * Accuracy, precision, recall, F1-score
   * Confusion matrices
   * Comparison between the four combinations:

     * LR + BoW
     * LR + TFIDF
     * MNB + BoW
     * MNB + TFIDF
   * Runtime measurements for each configuration

4. ### **Final Comparison**

   * A final discussion compares model performance to determine the most effective approach for this sentiment classification task.

---

## Getting Started

To run the notebook:

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Open the notebook:

   ```bash
   jupyter notebook Sentiment_Analysis.ipynb
   ```

---


## Purpose

This project is intended for educational and research purposes, showing how traditional machine learning models can still perform well on text classification tasks when combined with proper preprocessing and feature engineering.

---

## Author

Developed by \[Sebastiano Urzi']
Contact: \[[Nossebastian@gmail.com](mailto:Nossebastian@gmail.com)] 


