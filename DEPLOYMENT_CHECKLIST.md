# 🚀 Quick Deployment Checklist

## ✅ Files Created for Streamlit Deployment

- ✅ `streamlit_app.py` - Main Streamlit application
- ✅ `requirements.txt` - Updated with Streamlit
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `packages.txt` - System dependencies
- ✅ `.gitignore` - Git ignore rules
- ✅ `run_streamlit.sh` - Local test script
- ✅ `STREAMLIT_DEPLOYMENT.md` - Full deployment guide

## 🎯 Deploy to Streamlit Cloud (5 Steps)

### Step 1: Test Locally
```bash
./run_streamlit.sh
```
Visit http://localhost:8501

### Step 2: Initialize Git
```bash
git init
git add .
git commit -m "MovieSent Streamlit app ready for deployment"
```

### Step 3: Create GitHub Repo
- Go to https://github.com/new
- Name: `moviesent-sentiment-analysis`
- Don't initialize with README
- Create repository

### Step 4: Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/moviesent-sentiment-analysis.git
git branch -M main
git push -u origin main
```

### Step 5: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repo
5. Main file: `streamlit_app.py`
6. Click "Deploy!"

## ⚠️ Important Notes

### Large Model Files (>100MB)?
```bash
# Use Git LFS
brew install git-lfs
git lfs install
git lfs track "saved_models/*.pkl"
git lfs track "saved_models/*.h5"
git add .gitattributes
git add saved_models/
git commit -m "Add models with Git LFS"
git push
```

### Memory Issues?
- Free tier: 1GB RAM limit
- Optimize: Use `@st.cache_resource`
- Upgrade: Streamlit Cloud paid plan

## 📊 Your App URL
After deployment: `https://YOUR-APP-NAME.streamlit.app`

## 🔄 Update Deployed App
Just push to GitHub main branch - auto-deploys!

```bash
git add .
git commit -m "Update app"
git push
```

## 💡 Tips

1. **Test first**: Always test locally before deploying
2. **Check logs**: Monitor Streamlit Cloud logs for errors
3. **Model size**: Keep models under 500MB for best performance
4. **Cache wisely**: Use `@st.cache_resource` for model loading

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | Add to `requirements.txt` |
| Model not loading | Check Git LFS for large files |
| Out of memory | Optimize models or upgrade plan |
| NLTK data missing | Included in `streamlit_app.py` |

## 📞 Need Help?

- 📖 Full guide: `STREAMLIT_DEPLOYMENT.md`
- 🌐 Streamlit docs: https://docs.streamlit.io/
- 💬 Community: https://discuss.streamlit.io/

---

🎬 **You're ready to deploy!** Follow the 5 steps above.
