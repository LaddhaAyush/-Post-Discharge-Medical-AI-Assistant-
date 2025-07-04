# Deployment Guide for Render

## Prerequisites
1. GitHub repository with your code
2. Render account (free tier available)
3. Groq API key

## Step-by-Step Deployment

### Option 1: Using Render Dashboard (Recommended)

#### Deploy Backend API
1. **Login to Render**: Go to [render.com](https://render.com) and sign in
2. **Connect GitHub**: Link your GitHub account and select your repository
3. **Create Web Service**:
   - Click "New" → "Web Service"
   - Connect your repository
   - Configure:
     - **Name**: `post-discharge-ai-backend`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`
     - **Plan**: Free
4. **Add Environment Variables**:
   - `GROQ_API_KEY`: Your actual Groq API key
   - `PYTHON_VERSION`: `3.11.0`
5. **Deploy**: Click "Create Web Service"

#### Deploy Frontend (Streamlit)
1. **Create Another Web Service**:
   - **Name**: `post-discharge-ai-frontend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
2. **Add Environment Variables**:
   - `BACKEND_URL`: Copy the URL from your backend service (e.g., `https://post-discharge-ai-backend.onrender.com`)
   - `PYTHON_VERSION`: `3.11.0`
3. **Deploy**: Click "Create Web Service"

### Option 2: Using render.yaml (Infrastructure as Code)

1. **Push render.yaml**: Make sure `render.yaml` is in your repository root
2. **Create from Blueprint**:
   - In Render dashboard, click "New" → "Blueprint"
   - Connect your repository
   - Render will automatically create both services

### Option 3: Manual Docker Deployment

If you prefer Docker, create a Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p logs data

EXPOSE $PORT

CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "$PORT"]
```

## Environment Variables Needed

### Backend Service
- `GROQ_API_KEY`: Your Groq API key (required)
- `PORT`: Automatically set by Render
- `PYTHON_VERSION`: `3.11.0`

### Frontend Service  
- `BACKEND_URL`: URL of your backend service
- `PORT`: Automatically set by Render
- `PYTHON_VERSION`: `3.11.0`

## Post-Deployment Configuration

1. **Test Backend**: Visit `https://your-backend-url.onrender.com/docs` to see API documentation
2. **Test Frontend**: Visit `https://your-frontend-url.onrender.com` to use the web interface
3. **Check Logs**: Monitor deployment logs in Render dashboard
4. **Update Frontend**: Update `BACKEND_URL` environment variable in frontend service

## Important Notes

### Free Tier Limitations
- Services sleep after 15 minutes of inactivity
- First request after sleep may take 30-60 seconds (cold start)
- 750 hours/month limit per service

### Data Persistence
- Render's free tier doesn't persist files between deployments
- Patient data (`data/patients.json`) is included in your repository
- Logs directory is recreated on each deployment

### Security
- Never commit API keys to your repository
- Use Render's environment variables for sensitive data
- Enable HTTPS (automatic on Render)

## Troubleshooting

### Common Issues
1. **Build Fails**: Check requirements.txt for version conflicts
2. **Service Won't Start**: Verify start command and PORT environment variable
3. **API Connection Issues**: Ensure BACKEND_URL is correct in frontend
4. **Import Errors**: Make sure all dependencies are in requirements.txt

### Debugging Steps
1. Check deployment logs in Render dashboard
2. Verify environment variables are set correctly
3. Test backend API endpoints directly
4. Check if services are sleeping (free tier)

### Getting Logs
- **Backend Logs**: Render Dashboard → Backend Service → Logs
- **Frontend Logs**: Render Dashboard → Frontend Service → Logs

## Custom Domain (Optional)
1. Purchase domain from registrar
2. In Render dashboard: Service → Settings → Custom Domains
3. Add your domain and configure DNS

## Scaling (Paid Plans)
- Upgrade to paid plans for:
  - No sleep time
  - Faster builds
  - More resources
  - Custom domains
  - Private repositories
