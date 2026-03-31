# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Create Sample SMS Dataset
data = pd.DataFrame({
    'Message': [
        "Congratulations! You won a free ticket to Bahamas. Call now!",
        "Hey, are we meeting today for lunch?",
        "Urgent! Your account has been hacked. Click here to reset",
        "Can you send me the notes from yesterday's class?",
        "You have won $1000 cash! Claim now",
        "Don't forget the meeting at 10 AM",
        "Free entry in 2 a weekly competition to win FA Cup",
        "Are you coming to the party tonight?"
    ],
    'Label': [
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham'
    ]
})

print("Sample SMS Dataset:")
print(data)

# Step 3: Encode Labels
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})

# Step 4: Split Data
X = data['Message']
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert Text to Numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test_vec)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Continuous User Input for SMS Classification
print("\nSpam SMS Classifier – Enter messages to predict. Type 'exit' to quit.\n")

while True:
    message = input("Enter SMS message:\n")
    
    if message.strip().lower() == "exit":
        print("Exiting Spam SMS Classifier. Goodbye!")
        break
    
    # Convert message to vector
    message_vec = vectorizer.transform([message])
    
    # Predict
    prediction = model.predict(message_vec)
    
    # Output result
    if prediction[0] == 1:
        print("Prediction: SPAM ❌\n")
    else:
        print("Prediction: HAM ✅\n")