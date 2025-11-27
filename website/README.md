# Scarlet Sails Website Dashboard

Interactive web interface for the Scarlet Sails Algorithmic Trading System. Real-time metrics, model documentation, and API reference.

## 🚀 Quick Start

### Local Development

```bash
cd website/

# Option 1: Python (fastest)
python -m http.server 8000

# Option 2: Node.js
npx http-server

# Option 3: VS Code Live Server extension
# Right-click index.html → "Open with Live Server"
```

Then open: **http://localhost:8000**

## 📍 Dashboard Entry Points

### Main Pages

| Page | URL | Purpose |
|------|-----|----------|
| **Home** | `/index.html` | Landing page & navigation |
| **Dashboard** | `/dashboard.html` | **Main metrics & real-time data** |
| **Models** | `/models.html` | Trading strategy descriptions |
| **API Docs** | `/api.html` | Data service reference |

## 📁 Files Structure

### Core Files
- **`dashboard.html`** - Real-time metrics dashboard (Sharpe Ratio, Win Rate, Total Trades)
- **`data-service.js`** - GitHub data fetching module (330 lines)
- **`data-schema.md`** - Complete data binding specification

### UI Components
- **`index.html`** - Landing page
- **`models.html`** - Model listing page
- **`api.html`** - API documentation
- **`styles.css`** - Dark theme styling
- **`script.js`** - Interactive features

### Documentation
- **`README.md`** - This file
- **`DEPLOYMENT_GUIDE.md`** - Full deployment instructions (GitHub Pages, Vercel, Netlify, Docker)

## ✨ Features

✅ Real-time data binding to GitHub repository  
✅ Dark theme with responsive design  
✅ Zero fabrication principle - no fake data  
✅ Automatic model discovery  
✅ Smart caching (5-minute TTL)  
✅ Graceful error handling  
✅ API documentation included  
✅ Multiple deployment options  

## 🔧 Architecture

```
HTML Pages → data-service.js → GitHub Raw Content API → Repository Data
```

### Data-Service Methods

```javascript
await DataService.getSharpeRatio()      // Get latest Sharpe ratio
await DataService.getWinRate()          // Get win rate metric
await DataService.getTotalTrades()      // Get total trades
await DataService.listModels()          // List available models
await DataService.getModelCode(name)    // Get model source code
DataService.getCacheStats()              // Cache information
DataService.clearCache()                 // Manual cache clear
```

## 🌐 Deployment

### GitHub Pages (Recommended)
```bash
1. Settings → Pages
2. Select branch: main, directory: /website
3. Auto-deploy on push

URL: https://antihrist-star.github.io/ScArlet-Sails/
```

### Vercel (Automatic Deployments)
```bash
1. Connect repo on vercel.com
2. Root Directory: website
3. Deploy
```

### Netlify
```bash
1. Connect repo on netlify.com
2. Base directory: website
3. Deploy
```

### Docker
```bash
docker build -t scarlet-sails-dashboard .
docker run -p 80:80 scarlet-sails-dashboard
```

## 📊 Data Binding

### How It Works
1. Dashboard loads → calls `data-service.js`
2. Service fetches from public GitHub URLs
3. Real repository data displays on page
4. Missing data shows as `"-- (No data)"`

### Zero Fabrication Principle
- **Data exists** → Display real value
- **Data missing** → Display placeholder
- **Never** generate fake data

## 🔐 Security

✅ No API keys in code  
✅ Public GitHub URLs only  
✅ Read-only access  
✅ No credentials stored  
✅ Safe for public hosting  

## 📝 Testing

```javascript
// In browser console (F12):
await DataService.getSharpeRatio()
await DataService.getWinRate()
DataService.getCacheStats()
```

## 📖 Documentation

- **Data Schema** → `data-schema.md`
- **Deployment** → `DEPLOYMENT_GUIDE.md`
- **API Reference** → `/api.html`

## 📄 License

MIT License - See LICENSE in repository root

## 🔗 Related

- Main Repository: `https://github.com/Antihrist-star/ScArlet-Sails`
- Branch: `Antihrist-star-patch-1/website`
- Status: 22 commits ahead, 137 behind main
