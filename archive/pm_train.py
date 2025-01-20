import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import category_encoders as ce

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

import pickle

version = "v1"
MODEL_FILE = f"ipl_win_pred_{version}.bin"


def h2h(df):

    df = df.sort_values(by=["short_name", "start_date"])
    df["shifted_home_win"] = df.groupby("short_name")["output"].shift(1)
    df["h2h_home_win_ratio_last_7"] = (
        df.groupby("short_name")["shifted_home_win"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["shifted_runs"] = df.groupby("short_name")["home_runs"].shift(1)
    df["h2h_avg_runs_last_7"] = (
        df.groupby("short_name")["shifted_runs"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def gen_ELO(df):
    df = df.sort_values("start_date")  # Ensure time order

    # Dictionary storing current Elo ratings for each team
    current_ratings = {}

    def get_rating(team):
        # Default to 1500 if the team isn't in our dict yet
        return current_ratings.get(team, 1500)

    K = 80
    updated_rows = []
    for idx, row in df.iterrows():
        teamA = row["home_team"]
        teamB = row["away_team"]
        S_A = row["output"]  # 1 if A wins, 0 if B wins, 0.5 tie

        R_A = get_rating(teamA)
        R_B = get_rating(teamB)

        # Expected score for Team A
        E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))

        # Update both teams
        R_A_new = R_A + K * (S_A - E_A)
        R_B_new = R_B + K * ((1 - S_A) - (1 - E_A))

        # Store the new ratings
        current_ratings[teamA] = R_A_new
        current_ratings[teamB] = R_B_new

        # Optionally add to the row’s data
        row["elo_home_before"] = R_A
        row["elo_away_before"] = R_B
        row["elo_home_after"] = R_A_new
        row["elo_away_after"] = R_B_new
        row["elo_before_diff"] = R_A - R_B
        updated_rows.append(row)

    # Create a new DataFrame with updated rating info
    df_updated = pd.DataFrame(updated_rows)
    return df_updated


def shift_and_roll_mean(
    df, group_col, value_col, new_col_name, window=7, min_periods=1, fillna_val=0
):
    """
    For the given 'value_col', do:
      1) Shift by 1 within each group.
      2) Rolling mean with the specified window.
      3) Create a new column with the result.
    """

    shifted = df.groupby(group_col)[value_col].shift(1)

    rolled = (
        shifted.groupby(df[group_col])
        .rolling(window=window, min_periods=min_periods)
        .mean()
    )
    df[new_col_name] = rolled.reset_index(level=0, drop=True)

    # Fill NaN for first matches
    df[new_col_name] = df[new_col_name].fillna(fillna_val)

    return df


def generate_new(my_df, team):
    my_df = my_df.sort_values(by=[team, "start_date"])
    my_df = shift_and_roll_mean(
        my_df, team, "home_runs", f"{team}_avg_runs_scored_last_7"
    )
    my_df = shift_and_roll_mean(
        my_df, team, "away_runs", f"{team}_avg_runs_conceded_last_7"
    )
    my_df = shift_and_roll_mean(my_df, team, "output", f"{team}_win_ratio_last_7")
    my_df = shift_and_roll_mean(my_df, team, "run_rate", f"{team}_avg_run_rate_last_7")
    my_df = shift_and_roll_mean(
        my_df, team, "bowl_econ", f"{team}_avg_bowl_econ_last_7"
    )
    my_df = shift_and_roll_mean(
        my_df, team, "home_boundaries", f"{team}_avg_boundaries_scored_last_7"
    )
    my_df = shift_and_roll_mean(
        my_df, team, "away_boundaries", f"{team}_avg_boundaries_conceded_last_7"
    )
    return my_df


def diff_n_ratio(df):
    D = 1e-7  # small constant
    df["team_diff_avg_runs_scored_last_7"] = (
        df["home_team_avg_runs_scored_last_7"] - df["away_team_avg_runs_scored_last_7"]
    )
    df["team_diff_avg_runs_conceded_last_7"] = (
        df["home_team_avg_runs_conceded_last_7"]
        - df["away_team_avg_runs_conceded_last_7"]
    )
    df["team_ratio_avg_run_rate_last_7"] = (df["home_team_avg_run_rate_last_7"] + D) / (
        df["away_team_avg_run_rate_last_7"] + D
    )
    df["team_ratio_win_ratio_last_7"] = (df["home_team_win_ratio_last_7"] + D) / (
        df["away_team_win_ratio_last_7"] + D
    )
    df["team_ratio_avg_bowl_econ_last_7"] = (
        df["home_team_avg_bowl_econ_last_7"] + D
    ) / (df["away_team_avg_bowl_econ_last_7"] + D)
    df["team_ratio_avg_boundaries_scored_last_7"] = (
        df["home_team_avg_boundaries_scored_last_7"] + D
    ) / (df["away_team_avg_boundaries_scored_last_7"] + D)
    df["team_ratio_avg_boundaries_conceded_last_7"] = (
        df["home_team_avg_boundaries_conceded_last_7"] + D
    ) / (df["away_team_avg_boundaries_conceded_last_7"] + D)
    return df


def pre_process(df):

    mdf = df.copy()
    print(f" >> Data Pre processing .. ")

    mdf = mdf.sort_values("start_date")  # Ensure time order

    mdf["output"] = (mdf["home_team"] == mdf["winner"]).astype(int)
    mdf["toss_won"] = (mdf["home_team"] == mdf["toss_won"]).astype(int)

    mdf["run_rate"] = mdf["home_runs"] / mdf["home_overs"]
    mdf["bowl_econ"] = mdf["away_runs"] / mdf["away_overs"]

    mdf[["season_match", "day_night_play"]] = mdf.description.str.extract(
        r"^(.*?)\s*\(([^)]*)\)"
    )

    # Head to head data
    mdf = h2h(mdf)

    # ELO rating
    mdf = gen_ELO(mdf)

    inter_df = generate_new(mdf, "home_team")
    proc_df = generate_new(inter_df, "away_team")

    # Difference and Ratios
    proc_df = diff_n_ratio(proc_df)

    proc_df["season_weight"] = np.exp((proc_df["season"] - proc_df["season"].min()) / 2)

    # CLEANUP
    print(f" >> Data Cleaning .. ")

    # data leakage candidates

    cols_to_remove = [
        "result",
        "1st_inning_score",
        "2nd_inning_score",
        "home_score",
        "away_score",
        "home_runs",
        "away_runs",
        "home_wickets",
        "away_wickets",
        "pom",
        "points",
        "highlights",
        "shifted_home_win",
        "shifted_runs",
    ]
    proc_df.drop(columns=cols_to_remove, axis=1, inplace=True)
    sec_cols_to_remove = [
        "id",
        "description",
        "name",
        "venue_name",
        "match_days",
        "umpire1",
        "umpire2",
        "tv_umpire",
        "referee",
        "reserve_umpire",
    ]
    proc_df.drop(columns=sec_cols_to_remove, axis=1, inplace=True)
    ter_cols_to_remove = [
        "winner",
        "super_over",
        "home_overs",
        "away_overs",
        "home_boundaries",
        "away_boundaries",
        "run_rate",
        "bowl_econ",
    ]
    proc_df.drop(columns=ter_cols_to_remove, axis=1, inplace=True)

    other_cols_to_remove = [
        "elo_home_after",
        "elo_away_after",
        "elo_home_before",
        "elo_away_before",
        'home_team_avg_runs_scored_last_7', 'away_team_avg_runs_scored_last_7',
        'home_team_avg_runs_conceded_last_7', 'away_team_avg_runs_conceded_last_7',
        'home_team_avg_run_rate_last_7', 'away_team_avg_run_rate_last_7',
        'home_team_win_ratio_last_7', 'away_team_win_ratio_last_7',
        'home_team_avg_bowl_econ_last_7', 'away_team_avg_bowl_econ_last_7',
        'home_team_avg_boundaries_scored_last_7', 'away_team_avg_boundaries_scored_last_7',
        'home_team_avg_boundaries_conceded_last_7', 'away_team_avg_boundaries_conceded_last_7'
    ]
    proc_df.drop(columns=other_cols_to_remove, axis=1, inplace=True)

    # other NLP fields (to be removed unless advanced analytics needed)
    cat_cols_to_remove = [
        "short_name",
        "end_date",
        #"start_date",
        "home_playx1",
        "away_playx1",
        "away_key_batsman",
        "away_key_bowler",
    ]  # 'home_captain', 'away_captain', 'home_key_batsman', 'home_key_bowler',
    proc_df.drop(columns=cat_cols_to_remove, axis=1, inplace=True)
    print(f" ## Data Prep complete!")

    proc_df['toss_won'] = proc_df.toss_won.fillna('')
    proc_df['decision'] = proc_df.decision.fillna('')
    proc_df['day_night_play'] = proc_df.day_night_play.fillna('')
    proc_df['season_match'] = proc_df.season_match.fillna('')

    for c in ['season', 'h2h_home_win_ratio_last_7', 'h2h_avg_runs_last_7']:
        mean_c = proc_df[c].mean()
        proc_df[c] = proc_df[c].fillna(mean_c)

    cat_cols = [col  for col in proc_df.columns if proc_df[col].dtype == object]
    for col in cat_cols:
        proc_df[col] = (
            proc_df[col]
            .str.lower()
            .str.strip()
            .str.replace(r'\s+', '_', regex=True)
        )

    return proc_df


if __name__ == "__main__":

    print(f" >> Reading IPL seasons summary data. ")
    raw_df = pd.read_csv("../DATA/all_season_summary.csv")

    pre_df = pre_process(raw_df)

    # Since it needs to be a time aware split, shuffle=False .. which results in past matches only for training and future matches for predicting.
    processed_df = pre_df.sort_values('start_date')  # Ensure time order

    df_full_train, df_test = train_test_split(processed_df, test_size=.2, shuffle=False, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=.25, shuffle=False, random_state=1)
    print(f"Length of df_train:{len(df_train)},  df_val:{len(df_val)},  df_test:{len(df_test)}")

    y_train = df_train.pop('output')
    y_val = df_val.pop('output')
    y_test = df_test.pop('output')
    y_full_train = df_full_train.pop('output')

    cat_cols = [col  for col in processed_df.columns if processed_df[col].dtype == object]
    cat_cols.extend(['venue_id',  'season', 'toss_won']) 
    num_cols = list(set(df_train.columns) - set(cat_cols))

    categorical_features_small = ["day_night_play", "decision"]
    high_cardinality_features = list(set(cat_cols) - set(categorical_features_small))

    # Target Encoding for High Cardinality Features
    target_enc = ce.TargetEncoder(cols=high_cardinality_features)
    df_train_encoded = target_enc.fit_transform(df_train, y_train)
    df_val_encoded = target_enc.transform(df_val)

    df_train = df_train_encoded.reset_index(drop=True)
    df_val = df_val_encoded.reset_index(drop=True)


    # Create Pipelines for Numerical and Categorical Features

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),    # Fill missing values with mean
        ('scaler', StandardScaler())                    # Standardize numerical values
    ])

    # Categorical Transformer for One-Hot Encoding (for low cardinality categorical features)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

        # Column Transformer to apply transformations to different features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, categorical_features_small)
        ],
        remainder='drop'  # Drop any columns not specified
    )


    # Final model that has reasonable acc/auc for T20 IPL data.
    model = XGBClassifier(
        n_estimators=30,
        learning_rate=.01,
        random_state=42,
        eval_metric='logloss'
    )

    fold_aucs = []
    fold_accs = []
    tscv = TimeSeriesSplit(n_splits=5)

    for train_idx, val_idx in tscv.split(df_full_train):
        X_train, X_val = df_full_train.iloc[train_idx], df_full_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_full_train.iloc[train_idx], y_full_train.iloc[val_idx]

        # Transform with target encoding and preprocessor
        X_train_enc = target_enc.fit_transform(X_train, y_train_fold)
        X_val_enc   = target_enc.transform(X_val)
        

        pipeline = Pipeline([
            ('preprocessor', preprocessor),  
            ('model', model),
        ])
        pipeline.fit(X_train_enc, y_train_fold)

        y_pred_fold = pipeline.predict(X_val_enc)
        fold_accs.append(accuracy_score(y_val_fold, y_pred_fold))
        fold_aucs.append(roc_auc_score(y_val_fold, y_pred_fold))

    print(f"Accuracy={np.mean(fold_accs):.3f}, ROC AUC={np.mean(fold_aucs):.3f}")


    y_pred = pipeline.predict(df_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(f"Model {version} trained with Accuracy={acc:.3f}, ROC AUC={auc:.3f}")


    # Save Model

    with open(MODEL_FILE, "wb") as f_out:
        pickle.dump((target_enc, preprocessor, model), f_out)

    print(f"Output Model saved to : {MODEL_FILE}")

# NE=25|LR=0.01|MD=5|Subs=0.6|CS=0.8 Accuracy=0.555, ROC AUC=0.523
    # NE=25|LR=0.01|MD=7|Subs=0.6|CS=0.8 Accuracy=0.556, ROC AUC=0.526
# NE=30|LR=0.01|MD=4|Subs=0.6|CS=0.8 Accuracy=0.552, ROC AUC=0.522
# NE=30|LR=0.01|MD=5|Subs=0.6|CS=0.8 Accuracy=0.553, ROC AUC=0.525
# NE=30|LR=0.01|MD=7|Subs=0.6|CS=0.7 Accuracy=0.552, ROC AUC=0.521

# NE=100|LR=0.01|MD=20|Subs=0.7|CS=0.7 Accuracy=0.540, ROC AUC=0.526

# NE=50|LR=0.01|MD=20|Subs=0.7|CS=0.8 Accuracy=0.555, ROC AUC=0.533
