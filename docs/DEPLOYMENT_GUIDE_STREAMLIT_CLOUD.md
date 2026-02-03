# 🚀 Deployment Guide: Streamlit Cloud

This guide will help you deploy the **Solar Weather Monitoring System** to Streamlit Community Cloud.

## 📋 Prerequisites
- A **GitHub** account.
- This project pushed to a GitHub repository (already done!).
- A **Streamlit Cloud** account (sign up at [streamlit.io](https://streamlit.io/)).

## ⚙️ How it Works
We have configured the project to run in **"Simulation Mode"** by default.
- **Why?** The full AI models are >3GB. Downloading them on every app startup would be slow (10+ mins) and might crash the free cloud instance.
- **Result:** The app will deploy instantly and look 100% identical, using high-fidelity data simulations.

## 🚀 Deployment Steps

1.  **Log in** to [Streamlit Cloud](https://share.streamlit.io/).
2.  Click **"New app"**.
3.  **Connect to GitHub** (if not already connected).
4.  Select your repository: `Space-Weather-Monitor` (or whatever you named it).
5.  **Configure the deployment settings**:
    *   **Branch**: `main`
    *   **Main file path**: `dashboard.py`
6.  Click **"Deploy!"**.

## ☕ Wait for Deployment
Streamlit will now:
1.  Spin up a container.
2.  Install dependencies from `requirements.txt` (including our `Surya` engine).
3.  Launch `dashboard.py`.

Once complete, you will see your dashboard live! 🎈

## 🔄 Updating the App
Any time you `git push` changes to the `main` branch, Streamlit Cloud will automatically detect the update and re-deploy your app.

## 🛠️ Advanced: Enabling "Real Mode"
If you upgrade to a larger cloud instance or want to run the real models:
1.  Uncomment the model download lines in `download_models.py`.
2.  Update `dashboard.py` to call the real inference engine instead of `generate_forecast.py`.
