import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message']
df.head()
print(df.isnull().sum())

# Visualize the distribution of spam and ham messages
sns.countplot(x='label', data=df, palette='Set2')
plt.title('Spam vs Ham Message Distribution')
plt.show()

# Step 2: Text Preprocessing
# We need to convert text data to numerical format using TF-IDF Vectorizer

# Create a TF-IDF Vectorizer instance
vectorizer = TfidfVectorizer(stop_words='english')

# Transform the 'message' column to a tf-idf matrix
X = vectorizer.fit_transform(df['message'])

# Step 3: Split the data into training and testing sets
y = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary values (ham=0, spam=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Naive Bayes model
model = MultinomialNB()

# Fit the model on the training data
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example prediction on new data
example_text = ["Congratulations! You've won a lottery. Claim your prize now."]
example_vectorized = vectorizer.transform(example_text)
example_prediction = model.predict(example_vectorized)

print(f"Prediction for the given text: {'Spam' if example_prediction[0] == 1 else 'Ham'}")
