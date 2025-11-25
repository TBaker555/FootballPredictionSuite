# Football Prediction Suite

**Automated Football Predictions using Historical Data and Betting Odds**

This project predicts **Full-Time Result (FTR)** and **Both Teams To Score (BTTS)** outcomes for upcoming football fixtures using machine learning models (Random Forest and Logistic Regression) trained on historical football data.

---

## **Data Sources**

1. **Historical Data:** [Football-Data.co.uk](https://www.football-data.co.uk/) — provides match results and betting odds for past seasons.
2. **Fixture Data:** [Football-Data.co.uk](https://www.football-data.co.uk/) — provides upcoming match fixtures and odds.

> ⚠️ Users must ensure they have the **latest season data files and fixture files** for maximum efficiency and prediction accuracy.

---

## **Project Structure**

```
FootballPredictionSuite/
│
├── FolderMergeFiles.py      # Merge historical season Excel files into a single CSV
├── FTRPredictionsSept25.py  # Main prediction script
├── README.md
├── requirements.txt
├── .gitignore
├── Outputs/                 # (Optional) Folder for saved prediction CSVs
└── Predictions/             # (Optional) Folder for processed fixture predictions
```

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/FootballPredictionSuite.git
cd FootballPredictionSuite
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## **Usage**

1. **Merge historical data:**
   Run `FolderMergeFiles.py` to combine all season Excel files into one CSV.

```bash
python FolderMergeFiles.py
```

2. **Run predictions:**
   Update the path to your merged historical file and fixture file in `FTRPredictionsSept25.py`, then execute:

```bash
python FTRPredictionsSept25.py
```

3. **Outputs:**
   Predictions are saved as CSV files with timestamps for record-keeping in the project directory.

---

## **Notes & Recommendations**

- Predictions are **probabilistic** and based on historical data — results are **not guaranteed**.
- Always ensure your **season files and fixture files** are up-to-date for accurate predictions.
- For better model performance, you can **add new season data** to the merged CSV and retrain the models.
- Probability thresholds used in predictions (e.g., `BTTS_Prob >= 0.55`) can be adjusted for your preference.
- Attribution: Data sourced from [Football-Data.co.uk](https://www.football-data.co.uk/).

---

## **Sample Output**

| Date       | HomeTeam | AwayTeam | BTTS_Prob | BTTS_Pred | Prob_FTR_H | Prob_FTR_D | Prob_FTR_A | FTR_Pred |
| ---------- | -------- | -------- | --------- | --------- | ---------- | ---------- | ---------- | -------- |
| 2025-09-25 | TeamA    | TeamB    | 0.62      | 1         | 0.45       | 0.25       | 0.30       | H        |
| 2025-09-25 | TeamC    | TeamD    | 0.48      | 0         | 0.33       | 0.34       | 0.33       | D        |

> Columns:
>
> - `BTTS_Prob`: Probability both teams will score
> - `BTTS_Pred`: Prediction (1 = Yes, 0 = No)
> - `Prob_FTR_*`: Probability for Home Win / Draw / Away Win
> - `FTR_Pred`: Predicted full-time result

---
