# 🚀 OMIcare Fraud Detection App - Deployment Guide

## Overview
This guide provides multiple deployment options for your Streamlit fraud detection application. Choose the option that best fits your needs.

## 📋 Prerequisites
- Python 3.8+ installed locally
- Git repository (for cloud deployments)
- All project files present

---

## 🌟 Option 1: Streamlit Cloud (Recommended)

### Why Streamlit Cloud?
- ✅ **Free hosting**
- ✅ **Zero configuration**
- ✅ **Automatic deployments**
- ✅ **Built-in SSL**
- ✅ **Perfect for Streamlit apps**

### Steps:
1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `omicare_fraud_app.py`
   - Click "Deploy"

3. **Your app will be live at**: `https://your-app-name.streamlit.app`

---

## 🐳 Option 2: Docker Deployment

### Local Docker Testing:
```bash
# Build the image
docker build -t omicare-fraud-app .

# Run the container
docker run -p 8501:8501 omicare-fraud-app
```

### Docker Compose:
```bash
# Start with docker-compose
docker-compose up -d
```

### Deploy to Cloud with Docker:
- **Google Cloud Run**
- **AWS ECS/Fargate**
- **Azure Container Instances**
- **DigitalOcean App Platform**

---

## 🚂 Option 3: Railway Deployment

### Steps:
1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy**:
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Your app will be live at**: `https://your-app-name.railway.app`

---

## ☁️ Option 4: Heroku Deployment

### Steps:
1. **Install Heroku CLI**:
   - Download from [heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

2. **Login and Deploy**:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

3. **Your app will be live at**: `https://your-app-name.herokuapp.com`

---

## 🔧 Option 5: VPS/Server Deployment

### Using PM2 (Process Manager):
```bash
# Install PM2
npm install -g pm2

# Create ecosystem file
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'omicare-fraud-app',
    script: 'streamlit',
    args: 'run omicare_fraud_app.py --server.port=8501 --server.address=0.0.0.0',
    cwd: '/path/to/your/app',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production'
    }
  }]
}
EOF

# Start the app
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

### Using Nginx Reverse Proxy:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 🧪 Testing Your Deployment

### Local Testing:
```bash
# Test locally first
python run_app.py
# or
streamlit run omicare_fraud_app.py
```

### Verify Features:
- ✅ Login functionality
- ✅ Fraud Analyst Dashboard
- ✅ Claim submission
- ✅ File uploads
- ✅ PDF generation
- ✅ All data files accessible

---

## 🔒 Security Considerations

### For Production:
1. **Environment Variables**:
   ```bash
   # Set sensitive data as environment variables
   export OPENAI_API_KEY="your-api-key"
   export STREAMLIT_SERVER_HEADLESS=true
   ```

2. **Authentication** (Optional):
   - Add Streamlit authentication
   - Use OAuth providers
   - Implement session management

3. **HTTPS**:
   - Most cloud platforms provide HTTPS automatically
   - For VPS, use Let's Encrypt certificates

---

## 📊 Monitoring & Maintenance

### Health Checks:
- Streamlit provides built-in health checks at `/_stcore/health`
- Monitor application logs
- Set up uptime monitoring

### Updates:
- **Streamlit Cloud**: Automatic on git push
- **Docker**: Rebuild and redeploy
- **VPS**: Pull latest changes and restart

---

## 🆘 Troubleshooting

### Common Issues:

1. **Port Issues**:
   ```bash
   # Use environment variable for port
   streamlit run omicare_fraud_app.py --server.port=$PORT
   ```

2. **File Path Issues**:
   - Ensure all data files are included in deployment
   - Use absolute paths or relative paths from app root

3. **Memory Issues**:
   - Monitor memory usage
   - Consider upgrading server resources

4. **Dependencies**:
   - Ensure all packages in requirements.txt are compatible
   - Test locally before deploying

---

## 🎯 Recommended Deployment Strategy

### For Development/Testing:
- **Streamlit Cloud** (easiest, free)

### For Production:
- **Railway** or **Heroku** (reliable, scalable)
- **Docker on VPS** (full control)

### For Enterprise:
- **AWS/Azure/GCP** with Docker
- **Kubernetes** for high availability

---

## 📞 Support

If you encounter issues:
1. Check the logs in your deployment platform
2. Test locally first
3. Verify all files are included
4. Check environment variables
5. Review platform-specific documentation

---

**Choose your deployment method and follow the steps above. Your OMIcare Fraud Detection app will be live and accessible to users worldwide!**
