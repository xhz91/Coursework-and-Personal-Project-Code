import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

class Bow_Baseline_Model():
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.regressor = LogisticRegression()

    def train(self, train_df):
        """
        train_df (df): input training data
        Train the logistic regression model on bag-of-words representation.
        """
        # Identify feature and label columns from input df
        X_train = train_df["text"]
        y_train = train_df["label"]

        # Produce bag of word representation
        X_train = self.vectorizer.fit_transform(X_train.tolist())
        print(X_train)
        # Apply logistic regression on bag of word representation produced
        self.regressor.fit(X_train, y_train)
        
        return None

    def test(self, test_df):
        """
        test_df (df): input testign data
        Evaluate the trained model on a test set
        """
        # Identify feature and label columns from input df
        X_test_orig = test_df["text"]
        y_test = test_df["label"]

        # Produce bag of word representation
        X_test =  self.vectorizer.transform(X_test_orig.tolist())

        # Prediction and evaluation
        y_pred = self.regressor.predict(X_test)
        
        # Print examples of misclassification
        results_df = pd.DataFrame({
        "text": X_test_orig,  # Original text
        "true_label": y_test,  # Actual labels
        "predicted_label": y_pred  # Predicted labels
        })
        misclassify_eg = results_df[results_df["true_label"] != results_df["predicted_label"]].head()
        print(misclassify_eg)
        
        f1 = f1_score(y_test, y_pred)
        print(f"F1-score: {f1}")

        return f1



class Tfidf_Baseline_Model():
    def __init__(self):
        self.tfidfvectorizer = TfidfVectorizer()
        self.regressor = LogisticRegression()

    def train(self, train_df):
        """
        train_df (df): input training data
        Train the logistic regression model on bag-of-words representation.
        """
        # Identify feature and label columns from input df
        X_train = train_df["text"]
        y_train = train_df["label"]
        
        # Produce bag of word representation
        X_train = self.tfidfvectorizer.fit_transform(X_train.tolist())
        print(X_train)
        # Apply logistic regression on bag of word representation produced
        self.regressor.fit(X_train, y_train)
        
        return None

    def test(self, test_df):
        """
        test_df (df): input testign data
        Evaluate the trained model on a test set
        """
        # Identify feature and label columns from input df
        X_test_orig = test_df["text"]
        y_test = test_df["label"]

        # Produce TF-IDF representation
        X_test =  self.tfidfvectorizer.transform(X_test_orig.tolist())

        # Prediction and evaluation
        y_pred = self.regressor.predict(X_test)

        # Print examples of misclassification
        results_df = pd.DataFrame({
        "text": X_test_orig,  # Original text
        "true_label": y_test,  # Actual labels
        "predicted_label": y_pred  # Predicted labels
        })
        misclassify_eg = results_df[results_df["true_label"] != results_df["predicted_label"]].head()
        print(misclassify_eg)

        f1 = f1_score(y_test, y_pred)
        print(f"F1-score: {f1}")

        return f1
