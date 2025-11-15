"""
MovieSent: Dual Approach Sentiment Analysis - Gradio App
This app mirrors the Streamlit UI/UX but uses Gradio for quick deployment.
"""

import os
import re
import joblib
import numpy as np
import warnings
from functools import lru_cache

warnings.filterwarnings('ignore')

# Try to import optional heavy deps only when needed
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:
    nltk = None

# Download NLTK resources if available
if nltk is not None:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception:
        pass

# Preprocessing tools (if nltk is available)
if nltk is not None:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
else:
    stop_words = set()
    lemmatizer = None

# -----------------------------
# Text cleaning
# -----------------------------

def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    if nltk is not None and lemmatizer is not None:
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return ' '.join(tokens)
    else:
        return text

# -----------------------------
# Model loading with caching
# -----------------------------
@lru_cache(maxsize=1)
def load_models():
    """Return dict with lr_model, vectorizer, lstm_model (or None) and errors."""
    models = {
        'lr_model': None,
        'vectorizer': None,
        'lstm_model': None,
        'lstm_tokenizer': None,
        'lstm_params': None,
        'lstm_error': None,
    }

    try:
        models['lr_model'] = joblib.load('saved_models/lr_model.pkl')
        models['vectorizer'] = joblib.load('saved_models/vectorizer.pkl')
    except Exception as e:
        models['lstm_error'] = f'Error loading LR/vectorizer: {e}'
        return models

    # Load LSTM tokenizer and params (these are small and safe)
    try:
        models['lstm_tokenizer'] = joblib.load('saved_models/tokenizer.pkl')
        models['lstm_params'] = joblib.load('saved_models/lstm_params.pkl')
    except Exception:
        # It's okay if tokenizer/params not present; LSTM will be unavailable
        pass

    # Attempt to load LSTM model (try .h5 first, then .pkl)
    try:
        import tensorflow as tf
        h5_path = 'saved_models/lstm_model.h5'
        pkl_path = 'saved_models/lstm_model.pkl'

        if os.path.exists(h5_path):
            try:
                # Try loading with compile=False for compatibility
                models['lstm_model'] = tf.keras.models.load_model(h5_path, compile=False)
                try:
                    models['lstm_model'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                except Exception:
                    # If compile fails, ignore; model may still predict
                    pass
            except Exception as e_h5:
                # Try a softer compatibility approach: provide custom objects
                try:
                    from tensorflow.keras.layers import InputLayer

                    class CompatInputLayer(InputLayer):
                        def __init__(self, **kwargs):
                            if 'batch_shape' in kwargs:
                                batch_shape = kwargs.pop('batch_shape')
                                if isinstance(batch_shape, (list, tuple)) and len(batch_shape) > 1:
                                    kwargs['input_shape'] = tuple(batch_shape[1:])
                            super().__init__(**kwargs)

                    models['lstm_model'] = tf.keras.models.load_model(h5_path, custom_objects={'InputLayer': CompatInputLayer}, compile=False)
                    try:
                        models['lstm_model'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    except Exception:
                        pass
                except Exception as e2:
                    models['lstm_error'] = f'H5 loading failed: {e_h5} | fallback failed: {e2}'
        elif os.path.exists(pkl_path):
            try:
                models['lstm_model'] = joblib.load(pkl_path)
            except Exception as e_pkl:
                models['lstm_error'] = f'PKL loading failed: {e_pkl}'
        else:
            models['lstm_error'] = 'No LSTM model file found (.h5 or .pkl)'

    except Exception as e_tf:
        # TensorFlow not installed or other TF errors; mark LSTM unavailable
        models['lstm_error'] = f'TensorFlow unavailable or LSTM load error: {e_tf}'

    return models

# -----------------------------
# Prediction helpers
# -----------------------------

def predict_logistic_regression(review: str, models: dict):
    try:
        cleaned = clean_text(review)
        X = models['vectorizer'].transform([cleaned])
        pred = models['lr_model'].predict(X)[0]
        probs = models['lr_model'].predict_proba(X)[0]
        conf = float(max(probs)) * 100
        return {'sentiment': 'Positive' if pred == 1 else 'Negative', 'confidence': conf, 'error': None}
    except Exception as e:
        return {'sentiment': 'Error', 'confidence': 0.0, 'error': str(e)}


def predict_lstm(review: str, models: dict):
    if models.get('lstm_model') is None:
        return {'sentiment': 'Error', 'confidence': 0.0, 'error': models.get('lstm_error', 'LSTM not available')}

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        max_len = models.get('lstm_params', {}).get('max_len', 150)
        cleaned = clean_text(review)
        seq = models['lstm_tokenizer'].texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        probs = models['lstm_model'].predict(padded, verbose=0, batch_size=1)[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100
        return {'sentiment': 'Positive' if pred == 1 else 'Negative', 'confidence': conf, 'error': None}
    except Exception as e:
        return {'sentiment': 'Error', 'confidence': 0.0, 'error': str(e)}

# -----------------------------
# Gradio UI
# -----------------------------

def build_interface():
    try:
        import gradio as gr
    except Exception:
        raise RuntimeError('Gradio is not installed. Please install `gradio` in your environment.')

    models = load_models()

    with gr.Blocks(title='MovieSent - Sentiment Analysis') as demo:
        # Header
        gr.Markdown("# 🎬 MovieSent — Dual Approach Sentiment Analysis")
        gr.Markdown('Compare Logistic Regression and an optional LSTM model')

        with gr.Row():
            with gr.Column(scale=2):
                review_input = gr.Textbox(lines=6, placeholder='Type a movie review here...', label='Enter your movie review')
                with gr.Row():
                    pos_button = gr.Button('Positive Example')
                    neg_button = gr.Button('Negative Example')
                model_choice = gr.Radio(['Logistic Regression', 'LSTM', 'Both Models'], value='Both Models', label='Choose model')
                analyze_btn = gr.Button('Analyze Sentiment')

            with gr.Column(scale=1):
                gr.Markdown('## Model Status')
                if models.get('lstm_model') is not None:
                    gr.Markdown('**LSTM**: ✅ available')
                else:
                    gr.Markdown('**LSTM**: ⚠️ unavailable')
                    if models.get('lstm_error'):
                        with gr.Accordion("LSTM error details", open=False):
                            gr.Markdown(f"{models.get('lstm_error')}")
                gr.Markdown('**Logistic Regression**: ✅ available' if models.get('lr_model') is not None else '**Logistic Regression**: ✖️')

        # Result area
        with gr.Row():
            lr_output = gr.HTML(label='Logistic Regression Result')
            lstm_output = gr.HTML(label='LSTM Result')

        # Preprocessed text
        preproc = gr.Textbox(label='Preprocessed Text', interactive=False)

        # Example button handlers
        def set_positive():
            return ("This movie was absolutely amazing! The cinematography was breathtaking and the performances were outstanding. A must-watch masterpiece!",)

        def set_negative():
            return ("What a terrible waste of time. The plot was boring, the acting was wooden, and I couldn't wait for it to end. Very disappointing.",)

        pos_button.click(fn=set_positive, inputs=None, outputs=[review_input])
        neg_button.click(fn=set_negative, inputs=None, outputs=[review_input])

        # Analyze callback
        def analyze(review_text, choice):
            if not review_text or not review_text.strip():
                return ("", "", "")
            models_local = load_models()
            pre = clean_text(review_text)
            lr_html = ""
            lstm_html = ""

            if choice in ('Logistic Regression', 'Both Models'):
                lr_res = predict_logistic_regression(review_text, models_local)
                if lr_res['error']:
                    lr_html = f"<div style='padding:10px;border-radius:8px;background:#ffe6e6;color:#a00;'><strong>Error:</strong> {lr_res['error']}</div>"
                else:
                    color = '#d4edda' if lr_res['sentiment']=='Positive' else '#f8d7da'
                    icon = '😊' if lr_res['sentiment']=='Positive' else '😞'
                    lr_html = f"<div style='padding:10px;border-radius:8px;background:{color};'><h3 style='margin:0;'>{icon} {lr_res['sentiment']}</h3><p style='margin:0;'>Confidence: <strong>{lr_res['confidence']:.2f}%</strong></p></div>"

            if choice in ('LSTM', 'Both Models'):
                lstm_res = predict_lstm(review_text, models_local)
                if lstm_res['error']:
                    lstm_html = f"<div style='padding:10px;border-radius:8px;background:#fff3cd;color:#856404;'><strong>Note:</strong> {lstm_res['error']}</div>"
                else:
                    color = '#d4edda' if lstm_res['sentiment']=='Positive' else '#f8d7da'
                    icon = '😊' if lstm_res['sentiment']=='Positive' else '😞'
                    lstm_html = f"<div style='padding:10px;border-radius:8px;background:{color};'><h3 style='margin:0;'>{icon} {lstm_res['sentiment']}</h3><p style='margin:0;'>Confidence: <strong>{lstm_res['confidence']:.2f}%</strong></p></div>"

            return (lr_html, lstm_html, pre)

        analyze_btn.click(fn=analyze, inputs=[review_input, model_choice], outputs=[lr_output, lstm_output, preproc])

    return demo


if __name__ == '__main__':
    demo = build_interface()
    demo.launch(share=False, server_name='0.0.0.0', server_port=7860)
