# app_streamlit.py
import streamlit as st
import pickle
import joblib
import numpy as np
import re
from datetime import datetime
from nltk.tokenize import word_tokenize
import tensorflow.keras.preprocessing.text
import sys
import types
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ====================================
# NLTK Setup
# ====================================
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
except:
    pass

# ====================================
# COMPATIBILITY PATCH (Same as Flask)
# ====================================
try:
    import keras.src.legacy.preprocessing.text
except ImportError:
    m1 = types.ModuleType("keras.src")
    m2 = types.ModuleType("keras.src.legacy")
    m3 = types.ModuleType("keras.src.legacy.preprocessing")
    sys.modules["keras.src"] = m1
    sys.modules["keras.src.legacy"] = m2
    sys.modules["keras.src.legacy.preprocessing"] = m3
    sys.modules["keras.src.legacy.preprocessing.text"] = tensorflow.keras.preprocessing.text

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="MovieSent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================================
# CUSTOM CSS
# ====================================
st.markdown("""
<style>
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        min-height: 100vh;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    /* Header */
    .header-title {
        background: linear-gradient(135deg, #3b82f6 0%, #b9f2c9 50%, #c3ff00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        font-weight: 900;
        letter-spacing: -1px;
    }
    .header-subtitle {
        text-align: center;
        color: rgba(148, 163, 184, 0.9);
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.2);
    }
    .metric-number {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0;
    }
    .metric-label {
        color: rgba(148, 163, 184, 0.9);
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* Results */
    .result-positive {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(74, 222, 128, 0.1) 100%);
        border: 2px solid rgba(34, 197, 94, 0.5);
        color: #86efac;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(34, 197, 94, 0.15);
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(248, 113, 113, 0.1) 100%);
        border: 2px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(239, 68, 68, 0.15);
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .result-confidence {
        font-size: 3rem;
        font-weight: 900;
        margin-top: 1rem;
    }

    /* Input styling */
    textarea {
        border-radius: 12px !important;
        border: 2px solid rgba(100, 116, 139, 0.3) !important;
        background: rgba(15, 23, 42, 0.8) !important;
        color: white !important;
        font-size: 0.95rem !important;
    }
    textarea:focus {
        border-color: rgba(59, 130, 246, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 900;
        padding: 0.95rem 1.5rem;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease !important;
        font-size: 1.05rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.4) !important;
    }

    /* Info box */
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
    }

    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: rgba(226, 232, 240, 0.95) !important;
    }
    p, div {
        color: rgba(224, 224, 224);
    }
</style>
""", unsafe_allow_html=True)

# ====================================
# PERSISTENT USERS DB
# ====================================
USERS_DB_PATH = "saved_models/users_db.pkl"

def load_users_db_from_disk():
    if os.path.exists(USERS_DB_PATH):
        try:
            return joblib.load(USERS_DB_PATH)
        except:
            return {}
    return {}

def save_users_db_to_disk(users_db):
    os.makedirs(os.path.dirname(USERS_DB_PATH), exist_ok=True)
    joblib.dump(users_db, USERS_DB_PATH)

# ====================================
# SESSION STATE INIT
# ====================================
if "user" not in st.session_state:
    st.session_state.user = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users_db" not in st.session_state:
    st.session_state.users_db = load_users_db_from_disk()
if "active_page" not in st.session_state:
    st.session_state.active_page = "📊 Dashboard"

# ====================================
# LOAD MODELS
# ====================================
@st.cache_resource
def load_models():
    lr_model = joblib.load("saved_models/lr_model.pkl")
    vectorizer = joblib.load("saved_models/vectorizer.pkl")

    with open("saved_models/tokenizer.pkl", "rb") as f:
        lstm_tokenizer = pickle.load(f)

    with open("saved_models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("saved_models/lemmatizer.pkl", "rb") as f:
        lemmatizer = pickle.load(f)

    with open("saved_models/stopwords.pkl", "rb") as f:
        stop_words = pickle.load(f)

    try:
        lstm_model = load_model("saved_models/lstm_model.h5", compile=False)
    except:
        lstm_model = None

    return lr_model, vectorizer, lstm_tokenizer, label_encoder, lemmatizer, stop_words, lstm_model

try:
    lr_model, vectorizer, lstm_tokenizer, label_encoder, lemmatizer, stop_words, lstm_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")

# ====================================
# HELPERS
# ====================================
def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def normalize_sentiment(s):
    if not s:
        return "unknown"
    s = str(s).strip().lower()
    if s in ["positive", "pos", "1"]:
        return "positive"
    if s in ["negative", "neg", "0"]:
        return "negative"
    return "unknown"

def predict_logistic(review):
    try:
        cleaned = clean_text(review)
        transformed = vectorizer.transform([cleaned])
        pred = lr_model.predict(transformed)[0]
        prob = lr_model.predict_proba(transformed)[0]
        sentiment = label_encoder.inverse_transform([pred])[0]
        sentiment_norm = normalize_sentiment(sentiment)
        confidence = round(float(max(prob) * 100), 2)
        return sentiment_norm, confidence
    except:
        return "unknown", 0

def predict_lstm(review):
    try:
        if lstm_model is None:
            return "unknown", 0

        cleaned = clean_text(review)
        seq = lstm_tokenizer.texts_to_sequences([cleaned])
        if not seq or not seq[0]:
            return "unknown", 0

        padded = pad_sequences(seq, maxlen=100, padding="post", truncating="post")
        probs = lstm_model.predict(padded, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        sentiment = label_encoder.inverse_transform([pred_class])[0]
        sentiment_norm = normalize_sentiment(sentiment)
        confidence = round(float(probs[pred_class] * 100), 2)
        return sentiment_norm, confidence
    except:
        return "unknown", 0

def register_user(name, email, password):
    if email in st.session_state.users_db:
        return False, "Email already exists"
    st.session_state.users_db[email] = {
        "name": name,
        "password": password,
        "joined": datetime.now().strftime("%B %d, %Y"),
        "analyses": []
    }
    save_users_db_to_disk(st.session_state.users_db)
    return True, "Account created successfully!"

def login_user(email, password):
    if email not in st.session_state.users_db:
        return False, "Email not found"
    if st.session_state.users_db[email]["password"] != password:
        return False, "Invalid password"
    st.session_state.user = email
    st.session_state.logged_in = True
    st.session_state.active_page = "📊 Dashboard"
    return True, "Login successful!"

def logout_user():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.active_page = "📊 Dashboard"

# ====================================
# MAIN APP
# ====================================

# -----------------------------
# NOT LOGGED IN
# -----------------------------
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='header-title'>🎬 MovieSent</div>", unsafe_allow_html=True)
        st.markdown("<div class='header-subtitle'>Movie Review Sentiment Analysis Engine</div>", unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        auth_tab1, auth_tab2 = st.tabs(["Login", "Register"])

        with auth_tab1:
            st.markdown("### Welcome Back! 👋")
            login_email = st.text_input("Email Address", key="login_email", placeholder="you@example.com")
            login_password = st.text_input("Password", type="password", key="login_password", placeholder="••••••••••")
            if st.button(" Sign In", use_container_width=True, type="primary"):
                success, message = login_user(login_email, login_password)
                if success:
                    st.success(message)
                    st.balloons()
                    st.rerun()
                else:
                    st.error(message)

        with auth_tab2:
            st.markdown("### Create Your Account 🎉")
            reg_name = st.text_input("Full Name", key="reg_name", placeholder="John Doe")
            reg_email = st.text_input("Email Address", key="reg_email", placeholder="you@example.com")
            reg_password = st.text_input("Password", type="password", key="reg_password", placeholder="••••••••••")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm", placeholder="••••••••••")

            if st.button(" Create Account", use_container_width=True, type="primary"):
                if not all([reg_name, reg_email, reg_password, reg_confirm]):
                    st.error("❌ All fields are required!")
                elif reg_password != reg_confirm:
                    st.error("❌ Passwords do not match!")
                elif len(reg_password) < 6:
                    st.error("❌ Password must be at least 6 characters!")
                else:
                    success, message = register_user(reg_name, reg_email, reg_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

# -----------------------------
# LOGGED IN
# -----------------------------
else:
    if st.session_state.user and st.session_state.user in st.session_state.users_db:
        user_data = st.session_state.users_db[st.session_state.user]
        user_analyses = user_data.get("analyses", [])

        # NAVIGATION (Radio based)
        nav = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Analyze", "👤 Profile"],
            horizontal=True,
            label_visibility="collapsed",
            index=["📊 Dashboard", "🔍 Analyze", "👤 Profile"].index(st.session_state.active_page)
        )
        st.session_state.active_page = nav

        # ====================================
        # PAGE: DASHBOARD
        # ====================================
        if st.session_state.active_page == "📊 Dashboard":
            st.markdown(f"### 👋 Welcome {user_data['name']}! ")
            st.markdown("*Your sentiment analysis dashboard*")
            st.markdown("---")

            positive = sum(1 for a in user_analyses if normalize_sentiment(a.get("sentiment")) == "positive")
            negative = sum(1 for a in user_analyses if normalize_sentiment(a.get("sentiment")) == "negative")

            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📝</div>
                    <div class='metric-label'>Total Reviews</div>
                    <div class='metric-number'>{len(user_analyses)}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>😊</div>
                    <div class='metric-label'>Positive Reviews</div>
                    <div class='metric-number'>{positive}</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>😞</div>
                    <div class='metric-label'>Negative Reviews</div>
                    <div class='metric-number'>{negative}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            colA, colB, colC = st.columns([1, 2, 1])
            with colB:
                st.markdown("### 🚀 Ready to analyze?")
                if st.button("Start Analyzing Reviews →", use_container_width=True, type="primary"):
                    st.session_state.active_page = "🔍 Analyze"
                    st.rerun()

            if user_analyses:
                st.markdown("### 📋 Recent Analyses")
                for analysis in reversed(user_analyses[-5:]):
                    s = normalize_sentiment(analysis.get("sentiment"))
                    emoji = "😊" if s == "positive" else ("😞" if s == "negative" else "⚠️")
                    with st.expander(f"{emoji} {s.upper()} - {analysis.get('time')}"):
                        st.write(f"**Review:** {analysis.get('review','')[:200]}...")
                        cA, cB = st.columns(2)
                        with cA:
                            st.write(f"**Model:** {str(analysis.get('model','')).upper()}")
                        with cB:
                            st.write(f"**Confidence:** {analysis.get('confidence',0)}%")
            else:
                st.info("📊 No analyses yet! Head to Analyze to get started.")

        # ====================================
        # PAGE: ANALYZE
        # ====================================
        elif st.session_state.active_page == "🔍 Analyze":
            st.markdown("### 🎯 Analyze a Movie Review")
            st.markdown("Enter your movie review and let our AI analyze the sentiment")
            st.markdown("---")

            review = st.text_area(
                "Your Review:",
                placeholder="Write your thoughts about the movie...",
                height=160,
            )

            st.markdown("**Choose Analysis Model:**")
            model_choice = st.radio(
                "Model Option",
                ["⚡ Fast (Logistic Regression)", "🧠 Accurate (LSTM Neural Network)"],
                label_visibility="collapsed",
                index=0
            )
            model = "lr" if "Logistic" in model_choice else "lstm"

            if model == "lr":
                st.info("⚡ **Fast Model**: Uses Logistic Regression for quick, reliable predictions")
            else:
                st.info("🧠 **Accurate Model**: Uses LSTM Neural Network for deep learning predictions")

            st.markdown("---")

            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                analyze_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")
            with b2:
                clear_btn = st.button("🗑️ Clear", use_container_width=True)

            if analyze_btn:
                if not review.strip():
                    st.error("❌ Please enter a review to analyze!")
                else:
                    with st.spinner("🔄 Analyzing your review..."):
                        if model == "lstm":
                            sentiment_norm, confidence = predict_lstm(review)
                        else:
                            sentiment_norm, confidence = predict_logistic(review)

                    analysis_result = {
                        "review": review,
                        "sentiment": sentiment_norm,   # normalized stored
                        "confidence": confidence,
                        "model": model,
                        "time": datetime.now().strftime("%H:%M:%S")
                    }

                    user_data["analyses"].append(analysis_result)
                    st.session_state.users_db[st.session_state.user] = user_data
                    save_users_db_to_disk(st.session_state.users_db)

                    if sentiment_norm == "positive":
                        st.markdown(f"""
                        <div class='result-positive'>
                            <div class='result-title'>😊 POSITIVE SENTIMENT</div>
                            <div class='result-confidence'>{confidence}%</div>
                            <p style='margin-top: 1rem; opacity: 0.9;'>This review expresses positive sentiment</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("✅ Analysis completed and saved to your profile!")

                    elif sentiment_norm == "negative":
                        st.markdown(f"""
                        <div class='result-negative'>
                            <div class='result-title'>😞 NEGATIVE SENTIMENT</div>
                            <div class='result-confidence'>{confidence}%</div>
                            <p style='margin-top: 1rem; opacity: 0.9;'>This review expresses negative sentiment</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("✅ Analysis completed and saved to your profile!")

                    else:
                        st.warning("⚠️ Model could not confidently classify this review (Unknown). Try another review.")

            if clear_btn:
                st.rerun()

        # ====================================
        # PAGE: PROFILE
        # ====================================
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 👤 User Profile")
                st.markdown("---")
                st.write(f"**👤 Name:** {user_data['name']}")
                st.write(f"**📧 Email:** {st.session_state.user}")
                st.write(f"**📅 Member Since:** {user_data['joined']}")

            with col2:
                st.markdown("### 📊 Statistics")
                positive = sum(1 for a in user_analyses if normalize_sentiment(a.get("sentiment")) == "positive")
                negative = sum(1 for a in user_analyses if normalize_sentiment(a.get("sentiment")) == "negative")
                st.metric("Total Reviews", len(user_analyses))
                st.metric("Positive", positive)
                st.metric("Negative", negative)

            st.markdown("---")
            st.markdown("### 📚 About MovieSent")
            st.markdown("""
            **MovieSent** is a dual-model sentiment analysis engine:
            - ⚡ Logistic Regression (Fast)
            - 🧠 LSTM Neural Network (Accurate)
            """)

            st.markdown("---")
            if st.button("Logout", use_container_width=True):
                logout_user()
                st.rerun()

    else:
        st.error("❌ User not found. Please log in again.")
        logout_user()
        st.rerun()
