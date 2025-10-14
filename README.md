# ⚽ Football Analytics Dashboard

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://football-analytics-dashboard.streamlit.app/)

End‑to‑end xG modeling + Passing Networks + Match‑day Streamlit dashboard using **StatsBomb Open Data**.

## Quickstart (Demo in <10 minutes)

```bash
git clone <your-repo-url>
cd football-analytics-dashboard
python -m venv .venv && source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .[dev]
pre-commit install

# End‑to‑end demo (small sample)
python -m src.cli fetch --demo
python -m src.cli preprocess
python -m src.cli train --model xgb --calibration isotonic
python -m src.cli evaluate
streamlit run streamlit_app.py
```

## Full Data

```bash
python -m src.cli fetch --full
```

## Features

- **xG model**: LR baseline + XGBoost, calibrated (isotonic/Platt). Train/val split by match to avoid leakage. Metrics: ROC‑AUC, Brier, LogLoss. Reliability plot saved in `models/plots/`.
- **Passing networks**: Team‑match nodes from average locations; edges weighted by completed passes with threshold in UI.
- **Dashboard**: Match Overview (cumulative xG timeline), Shot Map & xG (interactive pitch), Passing Network (interactive network over pitch). Plotly uses `width="stretch"` (no deprecated `use_container_width`).

## Data Source

StatsBomb Open Data © StatsBomb – see their license and attribution. Data is fetched directly from GitHub raw files.

## Project Structure

```plaintext
football-analytics-dashboard/
├── data/ # raw/processed CSV (git‑ignored; .gitkeep)
├── models/ # joblib models + plots (git‑ignored)
├── notebooks/
├── src/
│ ├── config.py
│ ├── utils_io.py
│ ├── open_data.py
│ ├── preprocess_shots.py
│ ├── features_xg.py
│ ├── train_xg_model.py
│ ├── evaluate_xg.py
│ ├── passing_network.py
│ └── dashboard/
│ ├── __init__.py
│ ├── app.py
│ ├── components.py
│ ├── theming.py
│ ├── plots.py
│ └── pages/
│ ├── 1_Match_Overview.py
│ ├── 2_Shot_Map_&_xG.py
│ └── 3_Passing_Network.py
├── tests/
│ ├── test_features_xg.py
│ ├── test_preprocess_shots.py
│ ├── test_passing_network.py
│ └── test_cli.py
├── streamlit_app.py
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Configuration

See `src/config.py` – paths, caching flags. CLI via `python -m src.cli`.

## License & Attribution

This project uses **StatsBomb Open Data**. Please read and respect their license. This repo is for educational/portfolio use.

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
