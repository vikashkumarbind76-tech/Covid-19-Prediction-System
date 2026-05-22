# 🧬 COVID-19 Prediction System Deployment Guide

This guide provides instructions to deploy both the **Flask Web Application** and the **Streamlit AI Dashboard** in production environments.

---

## 📦 Containerized Deployment (Docker & Docker Compose)

Docker is the recommended deployment method for local staging or server-based hosting (AWS, GCP, DigitalOcean, Linode). It guarantees all packages, libraries, and SQLite connections compile correctly.

### Prerequisites
- Docker installed on your host system.
- Docker Compose installed on your host system.

### 1. Configure the Environment
Copy the example file to `.env`:
```bash
cp .env.example .env
```
Open `.env` and configure:
- `SECRET_KEY`: A secure session signing key.
- `ADMIN_PASSWORD`: Password for the admin dashboards.

### 2. Build and Start the Stack
From the project root directory, run:
```bash
docker-compose up --build -d
```

This will:
1. Build `Dockerfile.flask` and launch the Flask app on `http://localhost:5000`.
2. Build `Dockerfile.streamlit` and launch the Streamlit app on `http://localhost:8501`.
3. Create a persistent Docker volume named `covid_db_volume` mounted at `/app/data` to preserve database records across container restarts.

### 3. Check Status & Logs
```bash
# View running containers
docker-compose ps

# Monitor logs
docker-compose logs -f
```

### 4. Stop Services
```bash
docker-compose down
```

---

## ☁️ Cloud Deployments

If you prefer serverless or cloud application platforms, follow the guides below.

### 1. Flask Application (`api/app.py`)

#### A. Vercel (Serverless)
The project is preconfigured with a [vercel.json](file:///c:/Users/Vikash/OneDrive/Desktop/covid%20data%20copy/vercel.json) builder mapping.
1. Install Vercel CLI: `npm install -g vercel`
2. Run `vercel` from the root folder.
3. Configure environment variables in the Vercel dashboard:
   - `ADMIN_PASSWORD`
   - `SECRET_KEY`

> [!WARNING]
> Vercel functions are stateless and ephemeral. The SQLite database will reset on every function wakeup. To persist data in a production Vercel setup, consider replacing the SQLite driver in `api/app.py` with PostgreSQL (e.g. Supabase, Neon) or MongoDB.

#### B. Render or Railway (Persistent Containers)
Render and Railway support persistent disks which are ideal for SQLite databases.
1. Connect your Github repository.
2. Create a new **Web Service**.
3. **Command**: The platform will automatically run the [Procfile](file:///c:/Users/Vikash/OneDrive/Desktop/covid%20data%20copy/Procfile):
   `gunicorn api.app:app`
4. Set Environment Variables:
   - `PORT`: Automatically assigned by Render/Railway.
   - `ADMIN_PASSWORD`: Your admin password.
   - `SECRET_KEY`: A secure session key.
   - `DB_PATH`: `/var/data/predictions.db`
5. **Disk / Volume**: Add a persistent Disk/Volume to your web service and mount it to `/var/data`. This guarantees database records are never lost.

---

### 2. Streamlit Application (`app.py`)

#### A. Streamlit Community Cloud (Free Hosting)
1. Commit the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in.
3. Click **New app**, select your repository, branch, and set the entrypoint to `app.py`.
4. Click **Advanced settings** and set secrets:
   ```toml
   # Under Secrets
   DB_PATH = "covid_health.db"
   ```
5. Click **Deploy**.

#### B. Render / Railway
1. Create a new **Web Service**.
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Set Environment Variables:
   - `DB_PATH`: `/var/data/covid_health.db`
5. Mount a persistent Volume at `/var/data` to retain users and health records.

---

## 🔑 Environment Variables Reference

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | Flask: `5000`, Streamlit: `8501` | Server binding port. |
| `ADMIN_PASSWORD` | `Vikash09` | Access credentials for Flask and Streamlit admin sections. |
| `SECRET_KEY` | `covid19-prediction-secret` | Cryptographic signature for cookies/sessions. |
| `DB_PATH` | Flask: `predictions.db`, Streamlit: `covid_health.db` | Directory path pointing to SQLite files. |
