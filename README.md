# Spam SMS Classifier

## Description
This project is a Machine Learning-based application that classifies SMS messages as **Spam** or **Ham (Not Spam)**. It uses Natural Language Processing (NLP) techniques and a Naive Bayes classifier for accurate predictions.

## Features
- Text preprocessing using TF-IDF Vectorization
- Spam/Ham classification using Naive Bayes
- Interactive user input for real-time prediction
- Simple and beginner-friendly implementation

## Dataset
A sample dataset is created within the code containing labeled SMS messages:
- Spam messages (promotions, offers, scams)
- Ham messages (normal conversations)

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NLP (TF-IDF Vectorizer)

## Steps
1. Import libraries  
2. Create/load dataset  
3. Encode labels (spam = 1, ham = 0)  
4. Split dataset into training and testing  
5. Convert text to numerical vectors using TF-IDF  
6. Train Naive Bayes model  
7. Evaluate model (accuracy & confusion matrix)  
8. Take user input and predict spam/ham  

## How to Run
1. Install dependencies: