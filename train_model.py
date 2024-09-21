import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

nltk.download('stopwords')

# Initialize stopwords and stemmer
stop = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords and apply stemming
    text = ' '.join([ps.stem(word) for word in text.split() if word not in stop])
    return text

def main():
    # Define the path to the dataset 
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'imdb_reviews.csv')
    
    # Check if the dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset path '{dataset_path}' does not exist. Please check the path.")
    else:
        print(f"Dataset found at: {dataset_path}")
    
    print("Loading dataset...")
    data = pd.read_csv(dataset_path)
    
    print("Preprocessing text...")
    data['processed'] = data['review'].apply(preprocess_text)
    
    print("Vectorizing text...")
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['processed']).toarray()
    y = data['sentiment'].map({'positive': 1, 'negative': 0}).values
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the Multinomial Naive Bayes model...")
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    print("Predicting on test data...")
    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    print("Saving the trained model and vectorizer...")
    joblib.dump(classifier, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("Model and vectorizer saved successfully.")

if __name__ == '__main__':
    main()
