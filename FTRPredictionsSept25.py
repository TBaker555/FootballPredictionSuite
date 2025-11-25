import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from datetime import datetime

# Load historical merged data
df_hist = pd.read_csv("C://Users/HP/Desktop/MyPythonJouney/merged_seasons_data_latest.csv")
df_hist = df_hist.dropna(subset=["HomeTeam","AwayTeam","FTHG","FTAG","B365H","B365D","B365A","Avg>2.5"])
df_hist = df_hist.sort_values("Date").reset_index(drop=True)

# Compute BTTS, Over25, FTR
df_hist["BTTS"] = ((df_hist["FTHG"]>0) & (df_hist["FTAG"]>0)).astype(int)
df_hist["Over25"] = (df_hist["FTHG"]+df_hist["FTAG"]>2.5).astype(int)
df_hist["FTR"] = np.where(df_hist["FTHG"]>df_hist["FTAG"],"H", np.where(df_hist["FTHG"]<df_hist["FTAG"],"A","D"))

# Convert odds to probabilities
df_hist["Prob_H"] = 1/df_hist["B365H"]
df_hist["Prob_D"] = 1/df_hist["B365D"]
df_hist["Prob_A"] = 1/df_hist["B365A"]
df_hist["Prob_Over25"] = 1/df_hist["Avg>2.5"]
total_prob = df_hist["Prob_H"]+df_hist["Prob_D"]+df_hist["Prob_A"]
df_hist["Prob_H"]/=total_prob
df_hist["Prob_D"]/=total_prob
df_hist["Prob_A"]/=total_prob

# Helper functions
def compute_h2h(df,target_col,n_prev=10):
    h2h_dict = {}
    vals = []
    team_avgs = df.groupby('HomeTeam')[target_col].mean().to_dict()
    for idx,row in df.iterrows():
        teams=(row["HomeTeam"],row["AwayTeam"])
        val=np.mean(h2h_dict.get(teams,[team_avgs.get(row["HomeTeam"],0)]))
        vals.append(val)
        if teams not in h2h_dict:
            h2h_dict[teams]=[]
        h2h_dict[teams].append(row[target_col])
        if len(h2h_dict[teams])>n_prev:
            h2h_dict[teams].pop(0)
    return vals

def rolling3(df,team_col,target_col):
    df=df.sort_values("Date").copy()
    vals_list=[]
    roll_vals={}
    team_avgs=df.groupby(team_col)[target_col].mean().to_dict()
    for idx,row in df.iterrows():
        team=row[team_col]
        val=np.mean(roll_vals.get(team,[team_avgs.get(team,0)]))
        vals_list.append(val)
        if team not in roll_vals:
            roll_vals[team]=[]
        roll_vals[team].append(row[target_col])
        if len(roll_vals[team])>3:
            roll_vals[team].pop(0)
    return vals_list

# Feature engineering
df_hist["FTR_numeric"]=df_hist["FTR"].map({"H":1,"D":0,"A":-1})
df_hist["H2H_BTTS"]=compute_h2h(df_hist,"BTTS")
df_hist["H2H_FTR"]=compute_h2h(df_hist,"FTR_numeric")
df_hist["Home_BTTS_Last3"]=rolling3(df_hist,"HomeTeam","BTTS")
df_hist["Away_BTTS_Last3"]=rolling3(df_hist,"AwayTeam","BTTS")
df_hist["Home_FTR_Last3"]=rolling3(df_hist,"HomeTeam","FTR_numeric")
df_hist["Away_FTR_Last3"]=rolling3(df_hist,"AwayTeam","FTR_numeric")

feature_cols=["Prob_H","Prob_D","Prob_A","Prob_Over25","H2H_BTTS","H2H_FTR",
              "Home_BTTS_Last3","Away_BTTS_Last3","Home_FTR_Last3","Away_FTR_Last3"]

X=df_hist[feature_cols].fillna(df_hist[feature_cols].mean())
y_btts=df_hist["BTTS"]
y_ftr=df_hist["FTR"]

# Handle imbalance
smote=SMOTE(random_state=42)
X_btts_bal,y_btts_bal=smote.fit_resample(X,y_btts)
X_ftr_bal,y_ftr_bal=smote.fit_resample(X,y_ftr)

# Scale features
scaler_btts=StandardScaler()
X_btts_scaled=scaler_btts.fit_transform(X_btts_bal)
scaler_ftr=StandardScaler()
X_ftr_scaled=scaler_ftr.fit_transform(X_ftr_bal)

# Train models
rf_btts=RandomForestClassifier(n_estimators=500,max_depth=10,min_samples_leaf=5,min_samples_split=10,class_weight={0:1,1:1.5},random_state=42)
rf_btts.fit(X_btts_bal,y_btts_bal)
log_reg_btts=LogisticRegression(max_iter=1000,C=0.1,class_weight={0:1,1:1.5})
log_reg_btts.fit(X_btts_scaled,y_btts_bal)

rf_ftr=RandomForestClassifier(n_estimators=500,max_depth=10,min_samples_leaf=5,min_samples_split=10,class_weight='balanced',random_state=42)
rf_ftr.fit(X_ftr_bal,y_ftr_bal)
log_reg_ftr=LogisticRegression(max_iter=1000,C=0.1,class_weight='balanced',multi_class='ovr')
log_reg_ftr.fit(X_ftr_scaled,y_ftr_bal)

print("✅ Models trained. Predictions ready.")