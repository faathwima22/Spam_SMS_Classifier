# Spam_SMS_Classifier
A machine learning project to automatically detect spam messages from normal text messages using the Naive Bayes classifier.   Features: text preprocessing, TF-IDF vectorization, model training, and evaluation with accuracy and confusion matrix.   Dataset: SMS Spam Collection Dataset.
# Spam SMS Classifier

## Description
This project predicts whether a text message is **spam** or **not spam (ham)** using **Machine Learning**.  
We use the **Naive Bayes classifier**, which is very effective for text classification tasks.  

The goal is to automatically identify spam messages from normal messages, similar to how messaging apps filter spam.  

---

## Dataset
We use the **SMS Spam Collection Dataset**:  
[https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv)  

- The dataset contains **5,574 messages** labeled as `ham` (not spam) or `spam`.  
- Columns:  
  - `label` → `ham` or `spam`  
  - `message` → the text message content  

---

## Steps

1. **Import Libraries**  
   Use `pandas`, `scikit-learn` for ML and data processing.  

2. **Load Dataset**  
   Read dataset using pandas and explore first few rows.  

3. **Preprocess Data**  
   - Encode labels: `ham=0`, `spam=1`  
   - Check class distribution  

4. **Split Data**  
   - 80% training, 20% testing using `train_test_split`  

5. **Vectorize Text**  
   - Use `TfidfVectorizer` to convert text messages into numeric vectors  

6. **Train Model**  
   - Train `MultinomialNB` classifier on vectorized messages  

7. **Make Predictions**  
   - Predict on test set  
   - Test with custom messages  

8. **Evaluate Model**  
   - Accuracy score  
   - Confusion matrix  

---

## How to Run

1. Clone the repository to your local machine or open in **Google Colab**.  
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn