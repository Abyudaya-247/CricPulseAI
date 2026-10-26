# CricPulse AI
### Cricket Analytics & Win Probability Prediction

CricPulse AI is a cricket analytics and machine learning project inspired by live cricket score platforms. It analyzes historical IPL match data, visualizes batting and bowling performance, and estimates the probability of a chasing team winning based on the current match situation.

**Current status:** Historical-data prototype. Live score integration is planned for a future version.

## Key Features

- **Match analytics:** Match summaries, total runs, wickets, and innings comparisons.
- **Scoring progression:** Interactive charts for cumulative and over-by-over scoring.
- **Batter analytics:** Player-level run statistics and strike-rate analysis.
- **Bowler analytics:** Overs, runs conceded, wickets, and economy rate.
- **Win probability prediction:** Logistic regression using historical chase situations.
- **Model evaluation:** Accuracy, ROC-AUC, log loss, Brier score, and probability calibration diagnostics.
- **Chronological evaluation:** Tests the model on later seasons separately from earlier training seasons.
- **Automated testing:** Regression tests for delivery processing, feature generation, and match eligibility.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data analysis | Pandas |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib |
| Machine learning | Scikit-learn |
| Model persistence | Joblib |
| Testing | Pytest |
| Data source | Historical IPL JSON data from Cricsheet |

## Project Structure

```text
CricPulseAI/
├── app.py
├── src/
│   ├── inspect_match.py
│   ├── build_training_data.py
│   ├── train_model.py
│   └── evaluate_chronological.py
├── tests/
│   ├── test_inspect_match.py
│   └── test_training_data.py
├── data/
│   ├── raw/
│   │   └── ipl_json/
│   └── processed/
│       ├── sample_deliveries.csv
│       ├── innings_summary.csv
│       ├── chase_training_data.csv
│       ├── calibration_plot.png
│       └── chronological_evaluation.csv
├── models/
│   └── win_probability_model.joblib
├── requirements.txt
└── README.md
```

Generated files may differ depending on the current development stage. Raw datasets, generated CSV files, and trained model artifacts should not automatically be committed to GitHub; use Git LFS or external storage if you later decide to publish large assets.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CricPulseAI
```

Replace the placeholder with your repository URL after publishing it.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the execution-policy approach that worked in your development environment, or activate the environment using the VS Code Python interpreter selector.

### 3. Install dependencies

```powershell
python -m pip install pandas streamlit plotly scikit-learn joblib matplotlib pytest
```

For reproducible installations, maintain a `requirements.txt` file containing the dependencies and versions tested in your environment.

### 4. Obtain historical IPL data

Download the IPL JSON archive from the official Cricsheet downloads page:

https://cricsheet.org/downloads/

Extract the match JSON files into:

```text
data/raw/ipl_json/
```

### 5. Build the processed datasets

```powershell
python src/inspect_match.py
python src/build_training_data.py
```

### 6. Train and evaluate the model

```powershell
python src/train_model.py
python src/evaluate_chronological.py
```

### 7. Run the automated tests

```powershell
python -m pytest -v
```

The current development test suite contains 12 tests. Run the suite after making changes to verify that the tests still pass.

### 8. Launch the dashboard

```powershell
streamlit run app.py
```

Streamlit will display a local URL that you can open in your browser.

## Machine Learning Approach

The model estimates whether a chasing team wins using historical match-state features.

### Input features

- Target score
- Current score
- Runs required
- Wickets remaining
- Balls remaining
- Current run rate
- Required run rate

### Target variable

- `1`: The chasing team eventually won.
- `0`: The chasing team did not win.

The current baseline uses logistic regression with median imputation and standardization. A majority-prior classifier provides a simple comparison point.

The model is trained using historical delivery states, and train/test splitting is performed by match in the random evaluation. A separate chronological evaluation trains on seasons before 2025 and evaluates on seasons from 2025 onward.

## Preliminary Evaluation Results

The following results were observed during development. They are experimental results from the current dataset and should be reproduced after rebuilding the dataset and rerunning the scripts.

### Random match-level split

| Metric | Baseline | Logistic regression |
|---|---:|---:|
| Accuracy | 53.01% | 78.53% |
| ROC-AUC | 0.5000 | 0.8737 |
| Log loss | 0.6914 | 0.4405 |
| Brier score | 0.2491 | 0.1454 |

### Chronological evaluation

Training uses seasons before 2025; the test set contains the 2025 and 2026 seasons present in the downloaded archive.

| Metric | Baseline | Logistic regression |
|---|---:|---:|
| Accuracy | 55.90% | 79.57% |
| ROC-AUC | 0.5000 | 0.9063 |
| Log loss | 0.6893 | 0.4142 |
| Brier score | 0.2481 | 0.1359 |

The reported chronological test set contains 140 matches and 16,061 delivery states. These states are not independent observations, and the 2026 season may be incomplete.

The results are encouraging but do not establish production-level performance. Dataset eligibility, match outcomes, probability calibration, and generalization to future matches require further validation.

## Data Source and Attribution

Historical match data is sourced from Cricsheet:

https://cricsheet.org/

Review the provider's current terms and attribution requirements before redistributing source data or derived datasets.

## Current Limitations

- The project currently focuses on historical IPL data.
- Live score ingestion has not yet been implemented.
- The prediction model is an experimental baseline, not a production-grade forecasting system.
- The current data builder uses simplified eligibility rules for completed T20 chases.
- Probability calibration and performance across different match situations require further analysis.
- The chronological evaluation uses the available 2025–2026 data and should be repeated as new completed matches become available.

## Future Improvements

- Integrate a cricket data API for live score updates.
- Build an update pipeline for the current score and match state.
- Compare logistic regression with tree-based models.
- Improve data validation for interrupted matches, unusual results, and innings edge cases.
- Evaluate calibration and performance on later seasons.
- Add model interpretation and feature-effect visualizations.
- Deploy the Streamlit dashboard and automate testing.

## Author

Developed as a portfolio project demonstrating skills in Python, data analysis, data visualization, feature engineering, machine learning, model evaluation, and software testing.
