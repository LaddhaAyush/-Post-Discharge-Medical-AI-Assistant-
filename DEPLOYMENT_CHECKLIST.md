# Pre-Deployment Checklist

## ✅ Files to Verify Before Deployment

### Required Files
- [ ] `requirements.txt` - Updated with all dependencies including pydantic
- [ ] `runtime.txt` - Specifies Python 3.11.0
- [ ] `Procfile` - Process definitions for services
- [ ] `render.yaml` - Render service configuration
- [ ] `build.sh` - Build script (optional)
- [ ] `DEPLOYMENT_GUIDE.md` - Complete deployment instructions

### Code Updates
- [ ] `backend_api.py` - Updated to handle PORT environment variable
- [ ] `app.py` - Updated to use BACKEND_URL environment variable
- [ ] Environment variables properly configured

### Data Files
- [ ] `data/patients.json` - Patient data exists
- [ ] `data/nephro.txt` - Medical knowledge base exists
- [ ] `data/nephro_faiss.index` - FAISS vector index exists

### API Keys
- [ ] Groq API key ready (don't commit to repo)
- [ ] Environment variables configured in Render

## 🚀 Deployment Steps Summary

### 1. Push to GitHub
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Deploy Backend on Render
1. New Web Service
2. Connect GitHub repository  
3. Configure:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`
   - Environment: Add `GROQ_API_KEY`

### 3. Deploy Frontend on Render
1. New Web Service
2. Same repository
3. Configure:
   - Build: `pip install -r requirements.txt`
   - Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Environment: Add `BACKEND_URL` (from backend service)

### 4. Test Deployment
- [ ] Backend API docs: `https://your-backend.onrender.com/docs`
- [ ] Frontend app: `https://your-frontend.onrender.com`
- [ ] Test patient lookup: Try "John Smith"
- [ ] Test conversation flow
- [ ] Test agent handoff

## 🔧 Common Issues & Solutions

### Build Failures
- Check Python version compatibility
- Verify all imports are in requirements.txt
- Check for typos in requirements.txt

### Runtime Errors
- Verify environment variables are set
- Check logs in Render dashboard
- Ensure PORT variable is handled correctly

### Connection Issues
- Verify BACKEND_URL in frontend environment
- Check if services are sleeping (free tier)
- Ensure CORS is properly configured

## 📋 Environment Variables

### Backend Service
```
GROQ_API_KEY=your_actual_groq_api_key
PYTHON_VERSION=3.11.0
```

### Frontend Service
```
BACKEND_URL=your-backend-service-url.onrender.com
PYTHON_VERSION=3.11.0
```

## 🎯 Success Criteria
- [ ] Backend responds to health check: `/health`
- [ ] Frontend loads without errors
- [ ] Patient data loads correctly
- [ ] Name extraction works
- [ ] Conversation flows properly
- [ ] Agent handoff functions
- [ ] Session management works
- [ ] Logs are generated

## 📞 Support URLs
- **Render Docs**: https://render.com/docs
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/
- **Streamlit Deployment**: https://docs.streamlit.io/streamlit-community-cloud
