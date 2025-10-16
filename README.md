# ⚽ Football Analytics Dashboard

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://football-analytics-dashboard.streamlit.app/](https://football-xgcloud.streamlit.app/))

Professional multi-tab football analytics dashboard featuring **Expected Goals (xG) modeling** and **Passing Network analysis** using **StatsBomb Open Data**.

## 🚀 Quick Start (< 5 minutes)

### One-Command Setup

```bash
# Clone and setup
git clone https://github.com/itzmore-mph/football-analytics-dashboard.git
cd football-analytics-dashboard
make setup && source .venv/bin/activate  # Windows: .venv\Scripts\activate
make install

# Run demo pipeline and launch dashboard
make demo && make run
```

### Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/itzmore-mph/football-analytics-dashboard.git
cd football-analytics-dashboard

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -U pip
pip install -e .[dev]
pre-commit install

# 4. Run demo pipeline (fetch data, train model)
python -m src.cli demo

# 5. Launch dashboard
streamlit run streamlit_app.py
```

## 📋 Available Make Commands

```bash
make help      # Show all available commands
make setup     # Create virtual environment
make install   # Install dependencies and pre-commit hooks
make run       # Launch Streamlit dashboard
make test      # Run pytest tests
make lint      # Check code quality with ruff
make format    # Format code with black, isort, ruff
make clean     # Remove cache and build artifacts
make demo      # Run full demo pipeline (fetch → train → evaluate)
```

## ✨ Features

### 📊 Multi-Tab Dashboard
Professional Streamlit interface with 5 dedicated tabs:

1. **📊 Overview Tab**
   - Match selection with team-level xG metrics
   - Quick statistics (total shots, goals, conversion rate)
   - Cumulative xG timeline with interactive Plotly chart
   - Data freshness indicators

2. **🎯 xG Model & Pitch Tab**
   - Model performance metrics (ROC-AUC, Brier Score)
   - Interactive xG threshold slider for shot classification
   - Color-coded shot map on StatsBomb pitch (goals=green, high xG=yellow, low xG=red)
   - Model calibration plot display
   - Shot size scaled by xG value

3. **🔗 Passing Network Tab**
   - Team-level passing network visualization
   - Player positions from average touch locations
   - Edge thickness scaled by pass count
   - Adjustable minimum pass threshold filter
   - Network statistics (players, connections, max passes)

4. **📈 Statistics Tab**
   - Team-level aggregated statistics
   - Top 10 players by xG
   - Sortable data tables
   - CSV export functionality for further analysis

5. **⚙️ Settings Tab**
   - Cache management (clear data/model cache)
   - Data source information
   - About section with feature descriptions

### 🧠 xG Modeling
- **Models**: Logistic Regression baseline + XGBoost classifier
- **Calibration**: Isotonic regression / Platt (sigmoid) scaling
- **Features**: Distance, angle, body part, pressure, set-piece indicator
- **Validation**: Match-based train/val split to prevent data leakage
- **Metrics**: ROC-AUC, Brier Score, Log Loss
- **Plots**: Calibration curve saved to `models/plots/calibration.png`

### 🕸️ Passing Networks
- Average player positions from pass start/end locations
- Edges filtered by minimum pass count (configurable)
- Degree centrality calculation with NetworkX
- Visual encoding: node size ∝ touches, edge width ∝ pass count

## 🔧 CLI Commands

The project includes a comprehensive CLI built with Typer:

```bash
# Fetch demo data (4 matches)
python -m src.cli fetch --demo

# Fetch full dataset (100+ matches, requires internet)
python -m src.cli fetch --full

# Preprocess shots from cached events
python -m src.cli preprocess

# Train xG model
python -m src.cli train --model xgb --calibration isotonic

# Evaluate model and generate plots
python -m src.cli evaluate

# Run full demo pipeline
python -m src.cli demo

# Build passing network for a specific match
python -m src.cli build-passing <match_id>
```

## 📊 Dashboard Screenshots

### Overview Tab
*Match overview with xG metrics and cumulative timeline*
- Team xG comparison
- Shot statistics (total shots, goals, conversion %)
- Interactive timeline showing xG accumulation over match minutes

### xG Model & Pitch Tab
*Shot map with xG-based color coding and model calibration*
- Model performance metrics displayed
- Adjustable xG threshold slider
- Color-coded shots: 🟢 Goals | 🟡 High xG | 🔴 Low xG
- Model calibration curve

### Passing Network Tab
*Team passing connections visualized on pitch*
- Player positions from average locations
- Edge thickness proportional to pass count
- Adjustable minimum pass filter
- Network statistics

### Statistics Tab
*Detailed team and player statistics*
- Team aggregates (total xG, average per shot, goals, shots)
- Top 10 players by xG
- Export to CSV functionality

### Settings Tab
*Configuration and cache management*
- Clear data/model caches
- Data source information
- Feature documentation

## 📦 Data Source

**StatsBomb Open Data** © StatsBomb – see their [license and attribution](https://github.com/statsbomb/open-data).  
Data is fetched directly from GitHub raw files with automatic caching and graceful fallback to local samples.

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

## ⚙️ Configuration

Configuration is managed in `src/config.py` using Pydantic:

- **Paths**: `data/`, `models/`, `models/plots/`
- **Caching**: LRU cache for remote JSON fetches
- **Demo Size**: 4 matches by default
- All paths auto-created on import

## 🧪 Testing

```bash
# Run all tests
make test
# or
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Current test coverage: 5/5 passing
```

Tests included:
- Feature engineering validation
- Data preprocessing
- CLI imports
- Passing network structure
- Shot distance and angle calculations

## 🎨 Code Quality

The project uses modern Python tooling:

```bash
# Check code quality
make lint

# Auto-format code
make format
```

Tools configured:
- **ruff**: Fast linter (replaces flake8, pylint)
- **black**: Code formatter
- **isort**: Import sorter
- **pre-commit**: Git hooks for automatic checks

## 🐛 Known Limitations & Future Enhancements

**Current Limitations:**
- Demo dataset is small (4 matches) for quick testing
- No player tracking/event data beyond shots and passes
- Model trained on limited feature set (no shot type, assist context)
- Passing networks use simple average positions (not phase-specific)

**Potential Enhancements:**
- Add possession-adjusted metrics
- Include defensive actions (tackles, interceptions)
- Multi-match aggregation views
- Player comparison tools
- Advanced xG features (assist quality, defensive pressure)
- Real-time match simulation
- Export to interactive HTML reports

## 🔍 Troubleshooting

**Issue**: `ModuleNotFoundError` when running CLI  
**Solution**: Ensure you're running commands with `python -m src.cli` (not `python src/cli.py`)

**Issue**: Dashboard shows "No data/model artifacts found"  
**Solution**: Run `make demo` or `python -m src.cli demo` to generate demo data

**Issue**: Streamlit port already in use  
**Solution**: Stop existing Streamlit process or use: `streamlit run streamlit_app.py --server.port 8502`

**Issue**: Tests fail with import errors  
**Solution**: Reinstall package: `pip install -e .[dev]`

**Issue**: Pre-commit hooks fail  
**Solution**: Run `make format` to auto-fix formatting issues

## 📁 Project Structure

```
football-analytics-dashboard/
├── src/
│   ├── config.py              # Configuration and paths
│   ├── cli.py                 # Typer CLI commands
│   ├── open_data.py           # StatsBomb data fetching
│   ├── preprocess_shots.py    # Shot data preprocessing
│   ├── features_xg.py         # xG feature engineering
│   ├── train_xg_model.py      # Model training pipeline
│   ├── evaluate_xg.py         # Model evaluation
│   ├── passing_network.py     # Passing network generation
│   ├── utils_io.py            # I/O utilities
│   └── dashboard/
│       ├── app.py             # Main Streamlit app (multi-tab)
│       ├── components.py      # Reusable UI components
│       ├── plots.py           # Plotly chart functions
│       └── theming.py         # Dashboard theme config
├── tests/
│   ├── test_features_xg.py
│   ├── test_preprocess_shots.py
│   ├── test_passing_network.py
│   └── test_cli.py
├── data/                      # Generated CSV data (git-ignored)
├── models/                    # Trained models & plots (git-ignored)
├── sample_data/               # Fallback sample data
├── streamlit_app.py           # Streamlit entry point
├── pyproject.toml             # Project metadata & dependencies
├── Makefile                   # Development commands
├── .pre-commit-config.yaml    # Git hooks configuration
└── README.md

```

## 🎓 Technical Details

### xG Model Architecture
- **Base Models**: Logistic Regression | XGBoost Classifier
- **Hyperparameters**: 300 trees, max_depth=4, learning_rate=0.05
- **Cross-Validation**: 3-5 fold GroupKFold by match_id
- **Calibration**: Isotonic regression (default) or Platt scaling
- **Features**: 8 engineered features (distance, angle, body part, pressure, time, set-piece)

### Data Processing Pipeline
1. **Fetch**: Download StatsBomb events from GitHub
2. **Extract**: Parse shots and passes from event data
3. **Engineer**: Build shot features (geometry, context)
4. **Train**: Fit XGBoost with calibration on 75% data
5. **Evaluate**: Test on held-out 25%, plot calibration curve
6. **Serve**: Load trained model in Streamlit for inference

### Passing Network Methodology
- **Nodes**: Players with average (x,y) from pass starts and receptions
- **Edges**: Directed edges weighted by completed pass count
- **Filtering**: Minimum pass threshold (default: 3)
- **Metrics**: Degree centrality via NetworkX
- **Visualization**: mplsoccer pitch with custom styling

## 📜 License & Attribution

This project uses **StatsBomb Open Data** © StatsBomb.  
Please read and respect their [license](https://github.com/statsbomb/open-data).  
This repository is for educational and portfolio use.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and run tests (`make test`)
4. Format code (`make format`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📬 Contact

**👨‍💻 Moritz Philipp Haaf, BSc MA**  
**📩 Email:** [itzmore.dev@gmail.com](mailto:itzmore.dev@gmail.com)  
**🔗 GitHub:** [https://github.com/itzmore-mph](https://github.com/itzmore-mph)  
**🔗 LinkedIn:** [https://linkedin.com/in/moritz-philipp-haaf/](https://linkedin.com/in/moritz-philipp-haaf/)  

🚀 **If you found this useful, give this repo a ⭐ and share your feedback!**
