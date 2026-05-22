# 🚀 How to Run COVID-19 Prediction System

## Quick Start (Easiest Method)

### Option 1: Automatic Setup and Run
1. **Double-click** `SETUP_AND_RUN.cmd`
2. Wait for setup to complete
3. Server will start automatically
4. Open browser and go to: `http://127.0.0.1:5000`

### Option 2: Manual Setup

#### Step 1: Install Python
- Download Python 3.8+ from https://www.python.org/
- During installation, check "Add Python to PATH"

#### Step 2: Install Dependencies
Open Command Prompt in this folder and run:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Step 3: Run the Server
```cmd
python api\app.py
```

## 🌐 Accessing the Application

Once the server is running:

### Main Dashboard
- URL: `http://127.0.0.1:5000/`
- Features: Prediction engine, analytics, history

### Admin Panel
- URL: `http://127.0.0.1:5000/admin`
- Default Password: `Vikash09`
- Features: View all predictions, export data, manage records

### Reports
- URL: `http://127.0.0.1:5000/report/<id>`
- Access from admin panel or after making a prediction

## 🔑 Admin Credentials

- **Password**: `Vikash09`
- To change: Set environment variable `ADMIN_PASSWORD`

## ⚠️ Important Notes

1. **Always run through Flask** - Launch the web server using `SETUP_AND_RUN.cmd` or `python api\app.py` and navigate to the local URL. Do not open template HTML files directly in a web browser.
2. **Database**: SQLite database (`predictions.db`) will be created automatically.
3. **ML Model**: Ensure `covid19Model.pkl` exists in the root folder.


## 🛠️ Troubleshooting

### "Python not found"
- Install Python from https://www.python.org/
- Make sure to check "Add Python to PATH" during installation

### "Module not found" errors
- Run: `pip install -r requirements.txt`

### "Port already in use"
- Another application is using port 5000
- Stop other Flask apps or change port in `api\app.py`

### Admin page shows template errors
- Don't open `admin.html` directly in browser
- Always access through Flask: `http://127.0.0.1:5000/admin`

## 📁 Project Structure

```
covid data copy/
├── api/
│   ├── app.py              # Flask backend
│   └── templates/          # Jinja2 HTML templates served by Flask
├── app.py                  # Streamlit frontend & analytics app
├── covid19Model.pkl        # ML model
├── predictions.db          # Database (auto-created)
├── requirements.txt        # Python dependencies
├── SETUP_AND_RUN.cmd      # Quick start script
└── RUN_SERVER.cmd         # Run server (after setup)
```


## 🎨 Features

✅ AI-powered COVID-19 risk prediction
✅ Real-time analytics dashboard
✅ Prediction history tracking
✅ Admin panel with data export
✅ PDF/Word report generation
✅ Responsive design (mobile-friendly)
✅ Cyberpunk neon green theme

## 💡 Tips

- Keep the Command Prompt window open while using the app
- Press `Ctrl+C` in Command Prompt to stop the server
- Refresh browser if you make changes to HTML files
- Check Command Prompt for error messages

---

**Need Help?** Check the error messages in the Command Prompt window.
