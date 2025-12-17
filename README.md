# 🏥 Liver Disease AI Assistant

An intelligent web application that predicts liver disease risk using **Random Forest** and provides explainable AI insights using **SHAP** values. It also features an **AI Doctor** (powered by Google Gemini) to generate personalized medical reports.

## ✨ Features
- **Machine Learning:** Predicts liver disease risk with high accuracy.
- **Explainable AI (SHAP):** Visualizes *why* the model made a specific prediction.
- **Interactive Charts:** Compare patient data vs. population benchmarks using Radar Charts.
- **AI Medical Report:** Generates a professional doctor's report using Google Gemini.

## 🛠️ Tech Stack
- Python
- Streamlit
- Scikit-learn
- Plotly
- Google Gemini API

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kamolchaisiri/liver-disease-ai.git
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up API Key:**

    Create a folder named .streamlit in the root directory.

    Create a file .streamlit/secrets.toml.

    Add your key: GOOGLE_API_KEY = "YOUR_API_KEY"
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset
This project uses the Indian Liver Patient Dataset.

## 🚀 Project File Checklist
- Ensure your project folder contains the following files:
   
- app.py (Main application code)
   
- indian_liver_patient.csv (Dataset)
   
- requirements.txt (List of dependencies)
   
- .gitignore (Configuration to exclude system files/secrets)
   
- README.md (Project documentation)
   
(Note: The .streamlit/secrets.toml folder/file should exist locally for your API Key, but it will not be uploaded to GitHub).

## Ready to Upload?
If you are ready, you can run the following Git commands in your terminal:
```bash
git init
git add .
git commit -m "Initial commit - Liver AI App"
```
## (After this, follow the instructions provided on your GitHub repository page to push the code)
