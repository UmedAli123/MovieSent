# MovieSent - Streamlit Cloud Deployment Guide

This guide will walk you through deploying your MovieSent sentiment analysis app to Streamlit Cloud.

## 📋 Prerequisites

1. A GitHub account
2. Git installed on your computer
3. All model files in the `saved_models/` directory

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

1. **Initialize a Git repository** (if not already done):
   ```bash
   cd "/Users/datascientist/Downloads/MovieSent Dual Approach Sentiment Analysis/FInal_MovieSent_project/project"
   git init
   ```

2. **Create a `.gitignore` file**:
   ```bash
   echo "__pycache__/
   *.pyc
   .DS_Store
   .venv/
   venv/
   .env
   *.log" > .gitignore
   ```

3. **Add all files to Git**:
   ```bash
   git add .
   git commit -m "Initial commit - MovieSent Streamlit app"
   ```

### Step 2: Push to GitHub

1. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Name it `moviesent-sentiment-analysis` (or any name you prefer)
   - **DO NOT** initialize with README (we already have one)
   - Click "Create repository"

2. **Push your code to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/moviesent-sentiment-analysis.git
   git branch -M main
   git push -u origin main
   ```

   Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 3: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create a new app**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/moviesent-sentiment-analysis`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **Wait for deployment**:
   - Streamlit will install dependencies and launch your app
   - This may take 5-10 minutes on first deployment
   - You'll get a URL like: `https://YOUR_APP_NAME.streamlit.app`

## 📁 Required Files (Already Created)

✅ `streamlit_app.py` - Main Streamlit application
✅ `requirements.txt` - Python dependencies
✅ `packages.txt` - System dependencies
✅ `.streamlit/config.toml` - Streamlit configuration
✅ `saved_models/` - Your trained models (must be committed)

## ⚙️ Important Notes

### Model Files

⚠️ **CRITICAL**: Make sure all model files are committed to Git:
- `saved_models/lr_model.pkl`
- `saved_models/lstm_model.pkl`
- `saved_models/vectorizer.pkl`
- `saved_models/tokenizer.pkl`
- `saved_models/lstm_params.pkl`

If your model files are large (>100MB), you'll need to use **Git LFS**:

```bash
# Install Git LFS
brew install git-lfs  # macOS
git lfs install

# Track large files
git lfs track "saved_models/*.pkl"
git lfs track "saved_models/*.h5"
git add .gitattributes
git add saved_models/
git commit -m "Add model files with Git LFS"
git push
```

### Memory Limits

Streamlit Cloud free tier has 1GB RAM limit. If your app exceeds this:
- Use model quantization to reduce size
- Consider using lighter models
- Upgrade to Streamlit Cloud paid plan

### Testing Locally

Before deploying, test your Streamlit app locally:

```bash
# Install Streamlit
pip install streamlit

# Run the app
streamlit run streamlit_app.py
```

Visit http://localhost:8501 to test.

## 🔧 Troubleshooting

### Issue: "Module not found" error

**Solution**: Ensure all dependencies are in `requirements.txt`

### Issue: Model files not loading

**Solution**: 
1. Check that files exist in `saved_models/` directory
2. Verify files are committed to Git (`git status`)
3. Check file paths are relative to app root

### Issue: App crashes on startup

**Solution**:
1. Check Streamlit Cloud logs (click "Manage app" > "Logs")
2. Verify TensorFlow version compatibility
3. Ensure NLTK data downloads correctly

### Issue: Out of memory error

**Solution**:
1. Optimize model loading with `@st.cache_resource`
2. Reduce model size
3. Consider upgrading Streamlit Cloud plan

## 📊 Post-Deployment

### Monitor Your App

- **Logs**: Click "Manage app" > "Logs" to see real-time logs
- **Analytics**: Monitor usage in Streamlit Cloud dashboard
- **Updates**: Push to GitHub main branch to auto-deploy updates

### Custom Domain (Optional)

You can use a custom domain:
1. Go to app settings
2. Add custom domain
3. Update DNS records as instructed

### Share Your App

Your app URL will be: `https://YOUR_APP_NAME.streamlit.app`

Share this URL with anyone - no authentication needed!

## 🎯 Alternative: Run Locally with Streamlit

If you prefer not to deploy to the cloud:

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run streamlit_app.py
```

## 📞 Support

For issues:
- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Community: https://discuss.streamlit.io/
- GitHub Issues: Create an issue in your repository

---

🎬 **Happy Deploying!** 🚀

Your MovieSent app will be live and accessible to anyone with the URL!
