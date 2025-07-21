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

You can download it from Kaggle at the following link:

🔗 [IMDB Dataset of 50K Movie Reviews – Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Once downloaded, make sure to place the CSV file (usually named `IMDB Dataset.csv`) in the root directory of this project or update the path in the notebook accordingly.

---

## 1.0 Notebook – Sentiment Analysis with Logistic Regression (LR) and Multinomial Naive Bayes (MNB)

The first notebook (`Sentiment_Analysis.ipynb`) walks through the complete process of preparing the data and training classification models using two different vectorization techniques: **Bag of Words (BoW)** and **TF-IDF (Term Frequency-Inverse Document Frequency)**.

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


## 2.0 Notebook  – Sentiment Analysis with Linear Discriminant Analysis (LDA)

The second notebook (`Sentiment_Analysis_LDA.ipynb`) in this repository focuses on implementing **Sentiment Analysis** using **Linear Discriminant Analysis (LDA)** as the classification algorithm. While the overall text preprocessing pipeline is similar to the first notebook, several key adaptations have been made to better suit LDA, which requires **dense and low-dimensional data**.

###  Text Preprocessing and NLP

As in the first notebook, the text data undergoes extensive **Natural Language Processing (NLP)** to enhance data quality and consistency. The steps include:

* Stripping HTML tags
* Lowercasing all text
* Removing punctuation
* **Removing stop words** using the English stopword list from `nltk`
* **Tokenization** and **stemming**

Notably, in this notebook **duplicates are not removed**. This is an intentional choice to **preserve class balance** and avoid potential bias in the LDA training process.

###  Feature Extraction and Dimensionality Reduction

Given that **LDA requires dense input matrices**, the feature extraction step differs slightly from the first notebook:

* Both **Bag of Words (BoW)** and **TF-IDF** representations are limited to a **maximum of 50,000 features**, to keep the matrices computationally manageable.
* Unlike sparse matrix models (such as Logistic Regression or Naive Bayes), LDA does **not efficiently handle high-dimensional sparse data**, making this limitation essential.

Before applying LDA, the feature matrices are:

1. **Normalized using L2 Normalization**, ensuring that all samples have a unit norm.
2. **Reduced in dimensionality via Principal Component Analysis (PCA)**, which extracts the most significant components of the data. This step is critical for improving LDA's performance and interpretability.

###  Classification with LDA

After dimensionality reduction, the transformed data is passed to a **Linear Discriminant Analysis classifier**, which is trained and evaluated on both BoW and TF-IDF inputs. As in the previous notebook, the evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Runtime analysis

The results are then compared, providing insights into how well LDA performs in comparison to the models used in the first notebook, particularly when coupled with dimensionality reduction.

---

###  Summary of Key Differences from Notebook 1

| Aspect                   | Notebook 1 (LR & MNB)            | Notebook 2 (LDA)                 |
| ------------------------ | -------------------------------- | -------------------------------- |
| Duplicate Removal        | ✅ Yes                            | ❌ No (to preserve class balance) |
| Stop Word Removal        | Optional                         | ✅ Yes (NLTK stop words)          |
| Feature Limit            | No limit                         | ✅ Max 50,000 features            |
| Normalization            | ❌ No normalization               | ✅ L2 Normalization before PCA    |
| Dimensionality Reduction | ❌ Not applied                    | ✅ PCA applied before LDA         |
| Classifier Used          | Logistic Regression, Naive Bayes | Linear Discriminant Analysis     |
| Input Data Format        | Sparse (works well for LR/MNB)   | Dense (required by LDA)          |

---

##  Requirements

To run the notebook, you'll need the following Python packages:

* `Python 3.6 or higher`
* `numpy`
* `pandas`
* `scikit-learn`
* `nltk`
* `matplotlib` 
  
---
## Getting Started

To run the notebook:

1. Clone this repository:

   ```bash
   git clone https://github.com/Sebaurzi02/Sentiment-Analysis-NLP-Model-Comparison
   ```
2. Install required packages:
   
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:

   ```bash
   jupyter notebook Sentiment_Analysis.ipynb
   jupyter notebook  Sentiment_Analysis_LDA.ipynb
   ```

---


## Purpose

This project is intended for educational and research purposes, showing how traditional machine learning models can still perform well on text classification tasks when combined with proper preprocessing and feature engineering.

---

## Author

Developed by \[**Sebastiano Urzi'**]

Contact: \[[Nossebastian@gmail.com](mailto:Nossebastian@gmail.com)] 


