"""
✅ 1. Import Libraries
These are tools used to clean text, build models, and evaluate accuracy.

✅ 2. Download Stopwords
Words like "the", "is", "a" are removed from text because they don’t add useful meaning.

✅ 3. Mount Google Drive
This lets you load the dataset from your Google Drive into Colab.

✅ 4. Read Twitter Dataset
You load a CSV file that contains 1.6 million tweets with sentiment labels:

0 = Negative tweet

4 = Positive tweet → converted to 1

✅ 5. Take a Small Sample
Instead of using all 1.6 million tweets (which is slow), you take 10,000 for fast training.

✅ 6. Clean the Tweets
You remove:

URLs

Mentions (@user)

Hashtags (#word)

Punctuation

Common stopwords
And convert everything to lowercase.

✅ 7. Convert Text to Numbers
Using TF-IDF, you turn the cleaned text into a numerical format so that machine learning can understand it.
"""
✅ 8. Split the Data
Divide the data into:

Training set (80%) – to teach the model.

Testing set (20%) – to check how well it learned.

✅ 9. Train the KNN Model
It finds the 5 nearest similar tweets (neighbors) and decides if your tweet is likely positive or negative based on them.

✅ 10. Make Predictions
It predicts sentiments of unseen (test) tweets.

✅ 11. Evaluate Accuracy

"""

Code: 

# Step 1: Import necessary libraries
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Download stopwords from nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 3: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 4: Load dataset from your Drive
path = '/content/drive/My Drive/Colab Notebooks/ML/training.1600000.processed.noemoticon.csv'
df = pd.read_csv(path, encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = df[['text', 'target']]

# Step 5: Convert sentiment label (4 to 1)
df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

# Step 6: Sample a smaller dataset (10,000 tweets)
df = df.sample(10000, random_state=42)

# Step 7: Clean the tweet text
def clean(text):
    text = re.sub(r'http\S+|@\w+|#\w+|[^A-Za-z\s]', '', text)  # Remove URLs, mentions, hashtags, special chars
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    return ' '.join([w for w in words if w not in stop_words])

df['cleaned'] = df['text'].apply(clean)

# Step 8: Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = df['target']

# Step 9: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 11: Make predictions and evaluate
y_pred = knn.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
