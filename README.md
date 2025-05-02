🎧 Spotify Song Recommender System with End-to-End MLOps Integration 🚀

This repository hosts a full-fledged **Spotify Song Recommender System** built with modern **MLOps** principles. It leverages machine learning to recommend songs based on user preferences and song features, and integrates a complete pipeline—from data versioning to model deployment and dashboard visualization.

---

🔧 Tech Stack & Tools
Machine Learning**: Scikit-learn, Pandas, NumPy
MLOps Tools:
Git – Source code version control
DVC (Data Version Control)– Data, model, and pipeline versioning
FastAPI – Backend API for model inference
Streamlit– Interactive web dashboard for real-time song recommendation
Deployment: Docker + CI/CD ready

---

🚀 Features

* 🔍 **Song Recommendation Engine** based on audio features from Spotify API
* 🧠 **ML Pipeline** for data preprocessing, model training, evaluation, and versioning using DVC
* ⚙️ **FastAPI Endpoint** for real-time recommendations
* 📊 **Streamlit Dashboard** for interactive user experience
* 🗃️ **Model & Data Versioning** with DVC
* 🔁 **Reproducible Pipelines** with modular, script-based structure
* 🚢 **Deployment Ready** with Docker support and FastAPI backend

---

📁 Project Structure

```
├── data/                 # Raw and processed data
├── src/                  # Source code (training, preprocessing, utilities)
├── models/               # Trained ML models (tracked with DVC)
├── api/                  # FastAPI app for model inference
├── dashboard/            # Streamlit UI for recommendations
├── dvc.yaml              # Pipeline stages for DVC
├── requirements.txt
├── Dockerfile
└── README.md
```

---

🔄 ML Pipeline Stages (via DVC)

1. **Data Collection & Cleaning**
2. **Feature Engineering**
3. **Model Training**
4. **Model Evaluation**
5. **Model Serialization**
6. **API Integration with FastAPI**
7. **Dashboard Visualization with Streamlit**
8. **Deployment via Docker**

---

🌐 Live Demo

> Coming soon – Will be hosted on **Render / Heroku / AWS / GCP**
> *(Instructions for deployment included in the repo)*

---
✅ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/spotify-recommender-mlops.git
cd spotify-recommender-mlops

# Install dependencies
pip install -r requirements.txt

# Setup DVC
dvc pull  # Pull data and model
dvc repro # Re-run pipeline

# Run FastAPI
uvicorn api.main:app --reload

# Run Streamlit dashboard
streamlit run dashboard/app.py
```

📌 TODOs
* [ ] Add unit tests for model and API
* [ ] Set up CI/CD with GitHub Actions
* [ ] Enable cloud deployment

🧠 Author
Made with 💡 by \[Your Name]
Connect: yashika-pal  | palyashika641@gmail.com

