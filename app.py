from flask import Flask, render_template, request
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import os
 
app = Flask(__name__)

# Load the trained model and vectorizer
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sentiment_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vectorizer.pkl')

# Check if model files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Initialize stopwords and stemmer
stop = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([ps.stem(word) for word in text.split() if word not in stop])
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Retrieve user input from the form
            user_input = request.form.get('text', '').strip()
            
            if not user_input:
                # Handle empty input
                return render_template(
                    'result.html',
                    sentiment='N/A',
                    negative_proba=0,
                    positive_proba=0,
                    user_input='No input provided.'
                )
            
            # Preprocess the input text
            processed_text = preprocess_text(user_input)
            
            # Vectorize the processed text
            vect_text = vectorizer.transform([processed_text]).toarray()
            
            # Predict sentiment
            prediction = model.predict(vect_text)[0]
            
            # Get prediction probabilities
            proba = model.predict_proba(vect_text)[0].tolist()  # Convert to list for serialization
            
            # Split probabilities into negative and positive
            negative_proba = proba[0]
            positive_proba = proba[1]
            
            # Determine sentiment label
            sentiment = 'Positive' if prediction == 1 else 'Negative'
            
            # Debugging prints (visible in terminal)
            print(f"User Input: {user_input}")
            print(f"Sentiment: {sentiment}")
            print(f"Negative Probability: {negative_proba}")
            print(f"Positive Probability: {positive_proba}")
            
            # Render the result page with the analysis
            return render_template(
                'result.html',
                sentiment=sentiment,
                negative_proba=negative_proba,
                positive_proba=positive_proba,
                user_input=user_input
            )
        
        except Exception as e:
            # Log the exception and show an error message
            print(f"Error during sentiment analysis: {e}")
            return render_template(
                'result.html',
                sentiment='Error',
                negative_proba=0,
                positive_proba=0,
                user_input='An error occurred during analysis.'
            )
    
    # For GET requests, render the home page
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
