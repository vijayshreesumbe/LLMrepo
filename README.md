# Employee Sentiment & Engagement Analysis

This repository contains a starter notebook and script to analyze an unlabeled employee messages dataset (`test.csv`). The goal is to label sentiment, perform EDA, compute monthly employee sentiment scores, rank employees, detect flight-risk employees, and build a linear regression model to analyze sentiment trends.

How to use

1. Place `test.csv` in the repository root (`c:\Users\hp\Downloads\llmtask\test.csv`). The expected minimal columns are:
   - `employee` (or `sender`, `from`, etc.) — identifier for the employee
   - `message` (or `text`, `body`) — the message content
   - `date` (or `timestamp`) — date/time of the message (ISO format or parseable)

   If your CSV uses different column names, open `analysis.ipynb` or `runner.py` and map the column names accordingly in the `COLUMN_MAP` dictionary.

2. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Run the quick script to produce outputs (images and CSVs):

```powershell
python runner.py
```

Artifacts produced

- `analysis.ipynb` — main notebook with documented steps and code cells.
- `runner.py` — runnable end-to-end script that generates outputs and visualizations.
- `visualizations/` — folder where visual outputs are saved.
- `outputs/` — CSV outputs: labeled data, monthly scores, rankings, flight-risk list.

Notes

- The starter code uses NLTK VADER for sentiment labeling (fast, no API calls). If you prefer a transformer-based labeling, uncomment the relevant code in the notebook.
- If `test.csv` is missing or has different columns, the script will prompt what it expects and where to change column mappings.

Next steps

- Drop `test.csv` into the repo and run `python runner.py` or open `analysis.ipynb`.
- I can further adapt the notebook to use a transformer-based classifier if you provide the dataset and want higher-quality labels.
