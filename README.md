# Football Analytics Dashboard

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://football-analytics-portfolio.streamlit.app/)

## Features

- Expected goals (xG) model training and evaluation  
- Interactive Streamlit dashboard for match-day analytics  
- Data fetching & preprocessing pipeline  

## Project Structure

```plaintext
football-analytics-dashboard/
├── data/                         # Raw & processed datasets (not tracked in git)
├── models/                       # Trained ML models
├── src/
│   ├── fetch_passing_data.py     # Extract passing data from raw JSON
│   ├── fetch_shots_data.py       # Extract shot data from raw JSON
│   ├── fetch_statsbomb.py        # Fetch match data from StatsBomb
│   ├── open_data.py              # Helpers for working with StatsBomb open data
│   ├── passing_network.py        # Network analysis helpers
│   ├── preprocess_xG.py          # Clean and process shot data
│   ├── train_xG_model.py         # Train xG model using XGBoost
│   ├── ui.py                     # Streamlit entry point
│   └── validation.py             # Data validation utilities
├── tests/                        # Pytest suite for validation & plotting
│   ├── conftest.py
│   ├── test_plots.py
│   ├── test_ui_helpers.py
│   └── test_validation.py
├── streamlit_app.py              # Streamlit launcher (imports src.ui)
├── main.py                       # CLI wrapper for pipeline tasks
├── requirements.txt              # Python dependencies
├── setup.py                      # Editable install metadata
└── README.md                     # Project documentation (this file)
```

---

## 🚀 Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/football-analytics-dashboard.git
cd football-analytics-dashboard
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

ℹ️ `mplsoccer` and `matplotlib` are required for the Matplotlib-based passing network export.
The Streamlit app works with the core dependencies alone.

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
streamlit run streamlit_app.py
```

### 5️⃣ Run the Test Suite

```bash
pytest
```

### Troubleshooting

- **StatsBomb downloads fail behind a proxy**
– set `HTTP_PROXY`/`HTTPS_PROXY` environment variables before launching Streamlit or the pipeline.
- **No data displayed**
– click **Update Dashboard** to execute the local pipeline. Validation warnings for malformed rows are shown inline.
- **Optional plotting libs missing**
– install extra dependencies with `pip install -r requirements.txt`
(they are already listed under "Visualization / optional extras").

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

**👨‍💻 Moritz Philipp Haaf, BSc MA**  
**📩 Email:** [itzmore.dev@gmail.com](mailto:itzmore.dev@gmail.com)  
**🔗 GitHub:** [https://github.com/itzmore-mph](https://github.com/itzmore-mph)  
**🔗 LinkedIn:** [https://linkedin.com/in/moritz-philipp-haaf/](https://linkedin.com/in/moritz-philipp-haaf/)  

🚀 **If you found this useful, give this repo a ⭐ and share your feedback!**
