# MovieSent: Dual Approach Sentiment Analysis

A sophisticated movie review sentiment analysis application using both **Logistic Regression** and **LSTM Neural Networks** for accurate sentiment prediction.

![MovieSent Demo](https://img.shields.io/badge/Status-Working-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12-blue) ![Flask](https://img.shields.io/badge/Flask-3.0.3-red)

## 🎬 Features

- **Dual Model Approach**: Compare predictions from both traditional ML and deep learning models
- **Beautiful Web Interface**: Modern, responsive UI with user authentication
- **Real-time Predictions**: Instant sentiment analysis with confidence scores
- **Model Comparison**: Side-by-side comparison of different approaches
- **High Accuracy**: Trained on extensive movie review datasets

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Option 1: Streamlit App (Recommended ⭐)

1. **Run the quick start script**
   ```bash
   ./run_streamlit.sh
   ```

2. **Or manually:**
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```

3. **Open your browser**
   - Navigate to: **http://localhost:8501**

### Option 2: Flask App (Original)

1. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install flask scikit-learn tensorflow nltk joblib numpy
   ```

3. **Run the application**
   ```bash
   python app_lstm.py
   ```

4. **Open your browser**
   - Navigate to: **http://localhost:5000**

## 📁 Project Structure

```
MovieSent/
├── streamlit_app.py         # 🆕 Streamlit application (Recommended)
├── app_lstm.py              # Flask application (Original)
├── requirements.txt         # Python dependencies
├── run_streamlit.sh         # Quick start script
├── STREAMLIT_DEPLOYMENT.md  # Deployment guide
├── templates/
│   └── index.html          # Flask web interface
├── static/
│   ├── css/style.css       # Flask styling
│   └── js/script.js        # Flask frontend logic
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── saved_models/           # Trained models
│   ├── lr_model.pkl        # Logistic Regression model
│   ├── lstm_model.pkl      # LSTM model
│   ├── vectorizer.pkl      # TF-IDF vectorizer
│   ├── tokenizer.pkl       # LSTM tokenizer
│   └── lstm_params.pkl     # LSTM parameters
└── MovieSent_final.ipynb   # Original training notebook
```

## 🎯 How to Use

1. **Create Account**: Register with your email and password
2. **Login**: Access the sentiment analysis dashboard
3. **Enter Review**: Type or paste a movie review
4. **Select Model**: Choose between Logistic Regression or LSTM
5. **Analyze**: Get instant sentiment prediction with confidence score

## 🔧 Models

### Logistic Regression
- **Type**: Traditional Machine Learning
- **Features**: TF-IDF vectorization
- **Speed**: Very fast predictions
- **Accuracy**: High baseline performance

### LSTM Neural Network
- **Type**: Deep Learning (Recurrent Neural Network)
- **Features**: Sequential text processing
- **Speed**: Moderate prediction time
- **Accuracy**: Advanced pattern recognition

## � Technical Details

- **UI Framework**: Streamlit (Modern) / Flask (Original)
- **Frontend**: Streamlit Components / HTML5, CSS3, JavaScript
- **ML Libraries**: scikit-learn, TensorFlow/Keras
- **Text Processing**: NLTK
- **Models**: Joblib serialization

## ☁️ Cloud Deployment

Deploy to **Streamlit Cloud** for free hosting:

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your repository
4. Deploy `streamlit_app.py`

📖 **Full deployment guide**: See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md)

## 🔍 API Endpoints

- `GET /` - Web interface
- `POST /api/predict` - Sentiment prediction
- `GET /api/health` - Health check
- `GET /api/models-status` - Model availability

## 📊 Example Usage

```python
# Example API request
import requests

response = requests.post('http://localhost:5000/api/predict', 
                        json={
                            'review': 'This movie was absolutely amazing!',
                            'model': 'lstm'
                        })

result = response.json()
print(f"Sentiment: {result['predictions']['lstm']['sentiment']}")
print(f"Confidence: {result['predictions']['lstm']['confidence']:.1f}%")
```

## 🚫 Troubleshooting

### Common Issues:

1. **TensorFlow Installation**: If you encounter TensorFlow issues on macOS:
   ```bash
   pip install tensorflow==2.16.1
   ```

2. **NLTK Data**: If NLTK resources are missing:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. **Port Already in Use**:
   ```bash
   lsof -ti:5000 | xargs kill -9  # Kill process on port 5000
   ```

## 📞 Support

If you encounter any issues:
1. Check that all dependencies are installed correctly
2. Ensure Python 3.12+ is being used
3. Verify all model files are present in `saved_models/`

## 🎓 Academic Use

This project demonstrates:
- **Machine Learning**: Feature extraction and classification
- **Deep Learning**: Sequential modeling with RNNs
- **Web Development**: Full-stack application development
- **Model Deployment**: Production-ready ML service

---

**Created by**: Data Science Team  
**Version**: 1.0.0  
**Last Updated**: October 2025

---

🎬 **Happy Movie Review Analysis!** 🎭