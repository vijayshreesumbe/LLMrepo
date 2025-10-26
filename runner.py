"""
Runner script for Employee Sentiment & Engagement Analysis.
Drop `test.csv` in this folder and run `python runner.py`.

This script performs the following:
- Loads dataset (auto-detects common column names; configurable COLUMN_MAP)
- Labels each message with sentiment via NLTK VADER
- Saves labeled CSV to outputs/labeled_test.csv
- Performs EDA and saves plots to visualizations/
- Computes monthly employee sentiment scores and rankings
- Detects flight-risk employees (rolling 30-day count of negative messages >=4)
- Trains a simple linear regression to predict monthly sentiment score from features

"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sentiment: use NLTK VADER
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
except Exception:
    print("NLTK or VADER not installed. Please run: pip install -r requirements.txt")
    raise

# Paths
ROOT = Path(__file__).resolve().parent
VIS_DIR = ROOT / "visualizations"
OUT_DIR = ROOT / "outputs"
VIS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# Column mapping (change if your CSV has different column names)
COLUMN_MAP = {
    "employee": ["employee", "sender", "from", "author", "user"],
    "message": ["message", "text", "body", "content"],
    "date": ["date", "timestamp", "datetime", "time"]
}

import argparse

# Allow dataset path via env var DATA_PATH or command-line argument --data
parser = argparse.ArgumentParser(description='Run sentiment analysis pipeline')
parser.add_argument('--data', help='Path to CSV dataset (overrides default test.csv)', default=None)
args = parser.parse_args()

env_path = os.getenv('DATA_PATH')
if args.data:
    CSV_PATH = Path(args.data)
elif env_path:
    CSV_PATH = Path(env_path)
else:
    CSV_PATH = ROOT / 'test.csv'

if not CSV_PATH.exists():
    print(f"ERROR: dataset not found. Searched path: {CSV_PATH}")
    print("Please provide the dataset via --data or set DATA_PATH env var or place 'test.csv' in the folder:", ROOT)
    sys.exit(1)

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")

# Find column names
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive
    lowcols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lowcols:
            return lowcols[c.lower()]
    return None

emp_col = find_col(df, COLUMN_MAP['employee'])
msg_col = find_col(df, COLUMN_MAP['message'])
date_col = find_col(df, COLUMN_MAP['date'])

if not (emp_col and msg_col and date_col):
    print("Could not auto-detect required columns. Please update COLUMN_MAP in runner.py to map your column names.")
    print('Detected columns:', df.columns.tolist())
    sys.exit(1)

print(f"Using columns: employee={emp_col}, message={msg_col}, date={date_col}")

# Parse date
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
print('Parsed dates; null dates:', df[date_col].isna().sum())

# Drop rows with missing essential fields
df = df.dropna(subset=[emp_col, msg_col, date_col]).copy()

# Sentiment labeling with VADER
nltk.download('vader_lexicon')
sv = SentimentIntensityAnalyzer()

def label_vader(text):
    s = sv.polarity_scores(str(text))
    c = s['compound']
    if c >= 0.05:
        return 'Positive'
    elif c <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

print('Labeling sentiments (VADER)...')
df['sentiment'] = df[msg_col].apply(label_vader)

# Save labeled data
labeled_csv = OUT_DIR / 'labeled_test.csv'
df.to_csv(labeled_csv, index=False)
print('Saved labeled dataset to', labeled_csv)

# EDA: distribution of sentiments
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df, order=['Positive','Neutral','Negative'])
plt.title('Sentiment Distribution')
plt.tight_layout()
plt.savefig(VIS_DIR / 'sentiment_distribution.png')
plt.close()

# Messages over time
df['date_only'] = df[date_col].dt.date
msgs_per_day = df.groupby('date_only').size().rename('count').reset_index()
plt.figure(figsize=(10,4))
plt.plot(msgs_per_day['date_only'], msgs_per_day['count'])
plt.xticks(rotation=45)
plt.title('Messages per Day')
plt.tight_layout()
plt.savefig(VIS_DIR / 'messages_per_day.png')
plt.close()

# Monthly employee scoring
df['year_month'] = df[date_col].dt.to_period('M')
score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['sent_score'] = df['sentiment'].map(score_map)

monthly = df.groupby([emp_col, 'year_month'])['sent_score'].sum().reset_index()
monthly['year_month'] = monthly['year_month'].astype(str)
monthly.to_csv(OUT_DIR / 'monthly_scores.csv', index=False)
print('Saved monthly scores to outputs/monthly_scores.csv')

# Rankings per month
rankings = {}
for ym, grp in monthly.groupby('year_month'):
    g = grp.copy()
    g_pos_sorted = g.sort_values(['sent_score', emp_col], ascending=[False, True])
    g_neg_sorted = g.sort_values(['sent_score', emp_col], ascending=[True, True])
    top3_pos = g_pos_sorted.head(3)
    top3_neg = g_neg_sorted.head(3)
    rankings[ym] = {'top3_positive': top3_pos, 'top3_negative': top3_neg}
    # save tables
    top3_pos.to_csv(OUT_DIR / f'top3_positive_{ym}.csv', index=False)
    top3_neg.to_csv(OUT_DIR / f'top3_negative_{ym}.csv', index=False)
print('Saved top-3 rankings per month to outputs/')

# Flight risk detection: rolling 30-day count of negative messages >=4 per employee
# We'll operate on the per-message level: for each employee, sort by date and compute rolling window counts
print('Computing flight-risk employees (rolling 30-day window, >=4 negative messages)')

df_sorted = df.sort_values([emp_col, date_col]).copy()

# For each row, compute number of negative messages in past 30 days for that employee
neg_mask = (df_sorted['sentiment'] == 'Negative')

flight_flags = []

for emp, emp_df in df_sorted.groupby(emp_col):
    dates = emp_df[date_col].values
    negs = (emp_df['sentiment']=='Negative').astype(int).values
    n = len(emp_df)
    emp_flag = [False]*n
    # Use two-pointer sliding window
    left = 0
    neg_count = 0
    for right in range(n):
        if negs[right]==1:
            neg_count += 1
        # move left while window >30 days
        while left <= right and (dates[right] - dates[left]).astype('timedelta64[D]').astype(int) > 30:
            if negs[left]==1:
                neg_count -= 1
            left += 1
        if neg_count >= 4:
            emp_flag[right] = True
    flight_flags.extend(emp_flag)

df_sorted['flight_flag_row'] = flight_flags

# Now get unique employees flagged at any point
flight_emps = sorted(df_sorted.loc[df_sorted['flight_flag_row'], emp_col].unique())
pd.Series(flight_emps).to_csv(OUT_DIR / 'flight_risk_employees.csv', index=False, header=['employee'])
print('Saved flight-risk employee list to outputs/flight_risk_employees.csv')

# Save a visualization of flight-risk counts
fr_counts = df_sorted.groupby(emp_col)['flight_flag_row'].any().reset_index()
fr_counts.columns = [emp_col, 'ever_flagged']
fr_counts['ever_flagged'] = fr_counts['ever_flagged'].astype(int)
plt.figure(figsize=(8,4))
fr_counts['ever_flagged'].value_counts().sort_index().plot(kind='bar')
plt.xticks([0,1], ['Not Flagged','Flagged'])
plt.title('Number of Employees Flagged as Flight-Risk (ever)')
plt.tight_layout()
plt.savefig(VIS_DIR / 'flight_risk_counts.png')
plt.close()

# Predictive modeling: predict monthly sentiment score using simple features
print('Building linear regression model on monthly data...')
# Build features: message_count, avg_message_len, avg_word_count per (employee, month)
agg = df.groupby([emp_col, 'year_month']).agg(
    message_count=('sent_score','size'),
    avg_char_len=(msg_col, lambda x: x.str.len().mean()),
    avg_word_count=(msg_col, lambda x: x.str.split().map(len).mean()),
    target_score=('sent_score','sum')
).reset_index()

# Drop rows with NaN
agg = agg.dropna()

X = agg[['message_count','avg_char_len','avg_word_count']]
y = agg['target_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Model MSE: {mse:.4f}, R2: {r2:.4f}')

# Save model results and coefficient table
coef_df = pd.DataFrame({'feature': X.columns, 'coef': model.coef_})
coef_df.to_csv(OUT_DIR / 'linear_regression_coefs.csv', index=False)
with open(OUT_DIR / 'model_metrics.txt','w') as f:
    f.write(f'MSE: {mse}\nR2: {r2}\n')

# Save agg and predictions
agg_out = agg.copy()
agg_out['y_pred'] = model.predict(agg[['message_count','avg_char_len','avg_word_count']])
agg_out.to_csv(OUT_DIR / 'monthly_features_and_preds.csv', index=False)

print('All done. Check the outputs/ and visualizations/ folders.')
