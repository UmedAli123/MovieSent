"""
MovieSent: Dual Approach Sentiment Analysis - Flask Backend with Lazy TensorFlow Loading
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Initialize Flask app
app = Flask(__name__)

# Set static and template paths
app.config['TEMPLATE_FOLDER'] = 'templates'
app.config['STATIC_FOLDER'] = 'static'

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Global variables for models
lr_model = None
vectorizer = None
lstm_model = None
lstm_tokenizer = None
lstm_params = None
tensorflow_available = False

# Load models without TensorFlow first
print("Loading models...")
try:
    # Load Logistic Regression model first
    lr_model = joblib.load('saved_models/lr_model.pkl')
    vectorizer = joblib.load('saved_models/vectorizer.pkl')
    print("✓ Logistic Regression model loaded successfully!")
    
    # Load LSTM tokenizer and params (these don't need TensorFlow)
    lstm_tokenizer = joblib.load('saved_models/tokenizer.pkl')
    lstm_params = joblib.load('saved_models/lstm_params.pkl')
    print("✓ LSTM tokenizer and parameters loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Function to lazy-load TensorFlow and LSTM model
def load_lstm_model():
    global lstm_model, tensorflow_available
    
    if lstm_model is not None:
        return True
        
    try:
        print("Loading LSTM model (this may take a moment)...")
        
        # Load the pre-trained LSTM model from pickle (it's already a Keras model)
        lstm_model = joblib.load('saved_models/lstm_model.pkl')
        tensorflow_available = True
        
        # Suppress TensorFlow warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        print("✓ LSTM model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️ LSTM model loading failed: {e}")
        tensorflow_available = False
        return False

# ====================================
# PREPROCESSING FUNCTION
# ====================================
def clean_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ====================================
# PREDICTION FUNCTIONS
# ====================================
def predict_logistic_regression(review):
    """Predict sentiment using Logistic Regression"""
    if lr_model is None or vectorizer is None:
        return {
            'sentiment': 'Error',
            'confidence': 0,
            'error': 'Logistic Regression model not loaded'
        }
    
    try:
        cleaned = clean_text(review)
        tfidf_features = vectorizer.transform([cleaned])
        prediction = lr_model.predict(tfidf_features)[0]
        confidence = lr_model.predict_proba(tfidf_features)[0]
        
        return {
            'sentiment': 'Positive' if prediction == 1 else 'Negative',
            'confidence': float(max(confidence)) * 100,
            'error': None
        }
    except Exception as e:
        return {
            'sentiment': 'Error',
            'confidence': 0,
            'error': str(e)
        }

def predict_lstm(review):
    """Predict sentiment using LSTM"""
    # Try to load LSTM model if not already loaded
    if not load_lstm_model():
        return {
            'sentiment': 'Error',
            'confidence': 0,
            'error': 'LSTM model not available'
        }
    
    try:
        import numpy as np
        
        # Get max_len from parameters, default to 150 if not found
        max_len = lstm_params.get('max_len', 150) if lstm_params else 150
        
        cleaned = clean_text(review)
        sequence = lstm_tokenizer.texts_to_sequences([cleaned])
        
        # Pad sequences using the stored max_len parameter
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        
        # Make prediction - this returns probabilities for each class
        prediction_probs = lstm_model.predict(padded, verbose=0, batch_size=1)[0]
        
        # Use argmax to get the predicted class (0=negative, 1=positive)
        prediction_class = np.argmax(prediction_probs)
        confidence = float(prediction_probs[prediction_class]) * 100
        
        return {
            'sentiment': 'Positive' if prediction_class == 1 else 'Negative',
            'confidence': confidence,
            'error': None
        }
    except Exception as e:
        return {
            'sentiment': 'Error',
            'confidence': 0,
            'error': f'LSTM prediction failed: {str(e)}'
        }

# ====================================
# FLASK ROUTES
# ====================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        review = data.get('review', '').strip()
        model_choice = data.get('model', 'logistic_regression')
        
        if not review:
            return jsonify({'error': 'Please enter a review'}), 400
        
        result = {
            'review': review,
            'predictions': {}
        }
        
        if model_choice == 'logistic_regression' or model_choice == 'both':
            result['predictions']['logistic_regression'] = predict_logistic_regression(review)
        
        if model_choice == 'lstm' or model_choice == 'both':
            result['predictions']['lstm'] = predict_lstm(review)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'MovieSent API is running!'})

@app.route('/api/models-status', methods=['GET'])
def models_status():
    """Check if models are loaded"""
    # Try to load LSTM model for status check (but don't fail if it doesn't work)
    lstm_available = load_lstm_model() if not tensorflow_available else True
    
    return jsonify({
        'status': 'Models status',
        'lr_model': lr_model is not None,
        'lstm_model': lstm_available and lstm_model is not None,
        'tensorflow_available': tensorflow_available
    })

# ====================================
# ERROR HANDLERS
# ====================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ====================================
# RUN APPLICATION
# ====================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MovieSent - Dual Approach Sentiment Analysis")
    print("="*60)
    print("Models Available:")
    print(f"  • Logistic Regression: {'✓' if lr_model else '✗'}")
    print(f"  • LSTM Network: {'✓' if lstm_tokenizer else '✗'} (lazy-loaded)")
    print("="*60)
    print("Starting Flask application...")
    print("Visit http://localhost:5000 to use the application")
    print("="*60 + "\n")
    
    # Run with debug mode (set to False in production)
    app.run(debug=True, host='0.0.0.0', port=5000)