"""
MovieSent: Dual Approach Sentiment Analysis - Streamlit App
"""

import streamlit as st
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
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass

download_nltk_data()

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load models with caching
@st.cache_resource
def load_models():
    """Load all models and preprocessing tools"""
    try:
        import os
        
        # Load Logistic Regression model
        lr_model = joblib.load('saved_models/lr_model.pkl')
        vectorizer = joblib.load('saved_models/vectorizer.pkl')
        
        # Load LSTM tokenizer and params
        lstm_tokenizer = joblib.load('saved_models/tokenizer.pkl')
        lstm_params = joblib.load('saved_models/lstm_params.pkl')
        
        # Try loading LSTM model - handle different formats
        lstm_model = None
        lstm_error = None
        
        # Try loading from .h5 file first (Keras format)
        if os.path.exists('saved_models/lstm_model.h5'):
            try:
                from tensorflow import keras
                lstm_model = keras.models.load_model('saved_models/lstm_model.h5')
            except Exception as e:
                lstm_error = f"H5 loading failed: {str(e)}"
        
        # If .h5 failed or doesn't exist, try .pkl file
        if lstm_model is None and os.path.exists('saved_models/lstm_model.pkl'):
            try:
                import tensorflow as tf
                # Set custom object scope for legacy compatibility
                with tf.keras.utils.custom_object_scope({}):
                    lstm_model = joblib.load('saved_models/lstm_model.pkl')
            except Exception as e:
                lstm_error = f"PKL loading failed: {str(e)}"
        
        return {
            'lr_model': lr_model,
            'vectorizer': vectorizer,
            'lstm_model': lstm_model,
            'lstm_tokenizer': lstm_tokenizer,
            'lstm_params': lstm_params,
            'lstm_error': lstm_error,
            'success': True
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {'success': False, 'error': str(e)}

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
def predict_logistic_regression(review, models):
    """Predict sentiment using Logistic Regression"""
    try:
        cleaned = clean_text(review)
        tfidf_features = models['vectorizer'].transform([cleaned])
        prediction = models['lr_model'].predict(tfidf_features)[0]
        confidence = models['lr_model'].predict_proba(tfidf_features)[0]
        
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

def predict_lstm(review, models):
    """Predict sentiment using LSTM"""
    if models.get('lstm_model') is None:
        error_msg = models.get('lstm_error', 'LSTM model not available')
        return {
            'sentiment': 'Error',
            'confidence': 0,
            'error': error_msg
        }
    
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Get max_len from parameters
        max_len = models['lstm_params'].get('max_len', 150)
        
        cleaned = clean_text(review)
        sequence = models['lstm_tokenizer'].texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        
        # Make prediction
        prediction_probs = models['lstm_model'].predict(padded, verbose=0, batch_size=1)[0]
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
            'error': str(e)
        }

# ====================================
# STREAMLIT APP
# ====================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="MovieSent - Sentiment Analysis",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .positive {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
        }
        .negative {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🎬 MovieSent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dual Approach Sentiment Analysis</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.write("""
        **MovieSent** uses two powerful approaches for sentiment analysis:
        
        🤖 **Logistic Regression**
        - Traditional ML approach
        - Fast predictions
        - TF-IDF features
        
        🧠 **LSTM Neural Network**
        - Deep learning approach
        - Sequential processing
        - Advanced pattern recognition
        """)
        
        st.header("⚙️ Model Selection")
        model_choice = st.radio(
            "Choose your analysis method:",
            ["Logistic Regression", "LSTM", "Both Models"],
            index=2
        )
        
        st.header("ℹ️ How to Use")
        st.write("""
        1. Enter a movie review
        2. Select your preferred model
        3. Click 'Analyze Sentiment'
        4. View the prediction results
        """)
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    if not models['success']:
        st.error("Failed to load models. Please check if all model files are present in the 'saved_models' directory.")
        return
    
    # Show model status
    if models.get('lstm_model') is not None:
        st.success("✅ All models loaded successfully!")
    else:
        st.warning("⚠️ Logistic Regression loaded. LSTM model unavailable (using Logistic Regression only)")
        if models.get('lstm_error'):
            with st.expander("🔍 LSTM Error Details"):
                st.error(models['lstm_error'])
                st.info("**Tip:** The LSTM model may have compatibility issues. Using Logistic Regression for predictions.")
    
    # Main content
    st.header("Enter Your Movie Review")
    
    # Text input
    review = st.text_area(
        "Type or paste your review here:",
        height=150,
        placeholder="Example: This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout..."
    )
    
    # Example reviews
    with st.expander("📝 Try Example Reviews"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Positive Example"):
                st.session_state.example_review = "This movie was absolutely amazing! The cinematography was breathtaking and the performances were outstanding. A must-watch masterpiece!"
        with col2:
            if st.button("Negative Example"):
                st.session_state.example_review = "What a terrible waste of time. The plot was boring, the acting was wooden, and I couldn't wait for it to end. Very disappointing."
    
    # Use example if clicked
    if 'example_review' in st.session_state:
        review = st.session_state.example_review
        st.text_area("Type or paste your review here:", value=review, height=150, key="example_display")
        del st.session_state.example_review
    
    # Analyze button
    if st.button("🎯 Analyze Sentiment", type="primary", use_container_width=True):
        if not review.strip():
            st.warning("⚠️ Please enter a review to analyze.")
        else:
            with st.spinner("Analyzing sentiment..."):
                # Create results columns
                if model_choice == "Both Models":
                    col1, col2 = st.columns(2)
                    
                    # Logistic Regression
                    with col1:
                        st.subheader("🤖 Logistic Regression")
                        lr_result = predict_logistic_regression(review, models)
                        
                        if lr_result['error']:
                            st.error(f"Error: {lr_result['error']}")
                        else:
                            sentiment_class = "positive" if lr_result['sentiment'] == "Positive" else "negative"
                            st.markdown(f"""
                                <div class="prediction-box {sentiment_class}">
                                    <h2 style="margin:0;">{'😊 Positive' if lr_result['sentiment'] == 'Positive' else '😞 Negative'}</h2>
                                    <p style="font-size: 1.5rem; margin:0.5rem 0 0 0;">
                                        Confidence: <strong>{lr_result['confidence']:.2f}%</strong>
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Progress bar
                            st.progress(lr_result['confidence'] / 100)
                    
                    # LSTM
                    with col2:
                        st.subheader("🧠 LSTM Neural Network")
                        lstm_result = predict_lstm(review, models)
                        
                        if lstm_result['error']:
                            st.error(f"Error: {lstm_result['error']}")
                        else:
                            sentiment_class = "positive" if lstm_result['sentiment'] == "Positive" else "negative"
                            st.markdown(f"""
                                <div class="prediction-box {sentiment_class}">
                                    <h2 style="margin:0;">{'😊 Positive' if lstm_result['sentiment'] == 'Positive' else '😞 Negative'}</h2>
                                    <p style="font-size: 1.5rem; margin:0.5rem 0 0 0;">
                                        Confidence: <strong>{lstm_result['confidence']:.2f}%</strong>
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Progress bar
                            st.progress(lstm_result['confidence'] / 100)
                
                else:
                    # Single model
                    if model_choice == "Logistic Regression":
                        st.subheader("🤖 Logistic Regression Result")
                        result = predict_logistic_regression(review, models)
                    else:
                        st.subheader("🧠 LSTM Neural Network Result")
                        result = predict_lstm(review, models)
                    
                    if result['error']:
                        st.error(f"Error: {result['error']}")
                    else:
                        sentiment_class = "positive" if result['sentiment'] == "Positive" else "negative"
                        st.markdown(f"""
                            <div class="prediction-box {sentiment_class}">
                                <h2 style="margin:0;">{'😊 Positive' if result['sentiment'] == 'Positive' else '😞 Negative'}</h2>
                                <p style="font-size: 1.5rem; margin:0.5rem 0 0 0;">
                                    Confidence: <strong>{result['confidence']:.2f}%</strong>
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar
                        st.progress(result['confidence'] / 100)
                
                # Show cleaned text
                with st.expander("🔍 View Preprocessed Text"):
                    cleaned = clean_text(review)
                    st.code(cleaned, language=None)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p><strong>MovieSent v1.0</strong> | Dual Approach Sentiment Analysis</p>
            <p>Built with Streamlit • TensorFlow • scikit-learn</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
