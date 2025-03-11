# Football Analytics Portfolio: Expected Goals (xG) & Passing Networks

## Project Overview
This project applies **machine learning and data science** to football analytics, focusing on **expected goals (xG) modeling** and **passing network analysis**. The system fetches football event data, processes shot data, trains an xG model, and visualizes results using an **interactive Streamlit dashboard**.

### Features
- **Expected Goals (xG) Model** → Predict shot quality using **XGBoost**.
- **Passing Network Analysis** → Analyze team playstyles using **NetworkX**.
- **Interactive Dashboard** → Visualize xG and passing networks in **Streamlit**.
- **Automated Data Pipeline** → Fetch and process football event data.

---

## Project Structure
```plaintext
Portfolio/
│── data/                 # Raw & processed datasets
│   │── 15946.json        # Raw match event data
│   │── shots_data.csv    # Extracted shot data
│   │── processed_shots.csv # Cleaned data for xG model
│
│── models/               # Trained ML models
│   │── xgboost_xg_model.pkl  # Trained xG model
│
│── src/                  # Source code for the project
│   │── fetch_statsbomb.py   # Fetch match data from StatsBomb
│   │── fetch_shots_data.py  # Extract shot data from raw JSON
│   │── preprocess_xG.py     # Clean and process shot data
│   │── train_xG_model.py    # Train xG model using XGBoost
│   │── passing_network.py   # Analyze passing networks
│   │── dashboard.py         # Streamlit dashboard
│   │── requirements.txt     # Python dependencies
│
│── README.md             # Project documentation (this file)
│── .gitignore            # Ignore unnecessary files (e.g., .csv, .pkl)
```

---

## 🚀 Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/football-analytics-portfolio.git
cd football-analytics-portfolio
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r src/requirements.txt
```

---

## 🏆 Running the Project

### 1️⃣ Fetch Football Data
```bash
python src/fetch_statsbomb.py
```

### 2️⃣ Extract & Preprocess Shot Data
```bash
python src/fetch_shots_data.py
python src/preprocess_xG.py
```

### 3️⃣ Train xG Model
```bash
python src/train_xG_model.py
```

### 4️⃣ Run Interactive Dashboard
```bash
streamlit run src/dashboard.py
```

---

## 📊 Example Visualizations
### Expected Goals (xG) Shot Map
- Displays **shot locations** & predicted xG on a football pitch.
- Uses **mplsoccer** & **Plotly** for visualization.

### Passing Network Analysis
- Shows **team passing structures** using **NetworkX**.
- Identifies **key playmakers & team dynamics**.

---

## 🔥 Future Improvements
- ✅ Deploy xG model API for real-time predictions.
- ✅ Enhance passing network analysis with tactical insights.
- ✅ Improve UI with team colors & player details.

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit pull requests.

---

## 📬 Contact
**👨‍💻 Your Name**  
**📩 Email:** [moritz_haaf@outlook.com](mailto:moritz_haaf@outlook.com)  
**🔗 GitHub:** [https://github.com/itzmore-mph](https://github.com/itzmore-mph)  
**🔗 LinkedIn:** [https://linkedin.com/in/moritz-philipp-haaf/](https://linkedin.com/in/moritz-philipp-haaf/)  

🚀 **If you found this useful, give this repo a ⭐ and share your feedback!**