# %%
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

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

import pickle
import os

version = os.getenv("MODEL_VERSION", "v1")
model_dir = os.getenv("MODEL_DIR", "../ipl_infer/MODEL")

MODEL_FILE = os.path.join(model_dir, f"ipl_chase_pred_{version}.bin")


team_map = {
    "Kolkata Knight Riders": "KKR",
    "Mumbai Indians": "MI",
    "Rajasthan Royals": "RR",
    "Royal Challengers Bangalore": "RCB",
    "Royal Challengers Bengaluru": "RCB",
    "Chennai Super Kings": "CSK",
    "Kings XI Punjab": "PK",
    "Punjab Kings": "PK",
    "Sunrisers Hyderabad": "SRH",
    "Deccan Chargers": "SRH",
    "Delhi Daredevils": "DCA",
    "Delhi Capitals": "DCA",
    "Pune Warriors": "SG",
    "Rising Pune Supergiant": "SG",
    "Rising Pune Supergiants": "SG",
    "Lucknow Super Giants": "SG",
    "Gujarat Titans": "GT",
    "Gujarat Lions": "GT",
    "Kochi Tuskers Kerala": "KTK"
}

def prep_data():
    print(f"Reading matches and deliveries dataset ..")
    matches_df = pd.read_csv('DATA/matches.csv')
    deliveries_df = pd.read_csv('DATA/deliveries.csv')

    # Interested only in second innings for determining successful chase.
    second_innings = deliveries_df[deliveries_df.inning == 2]
    second_innings.head()

    print(f" Data set size: Second innings:{len(second_innings)}, Matches: {len(matches_df)}")
    

    gdf = second_innings.sort_values(['match_id', 'over', 'ball']).reset_index(drop=True)

    # Rolling aggregate of last 2 overs
    gdf['runs_last_12_balls'] = (
        gdf.groupby('match_id', group_keys=False)['total_runs']
        .apply(lambda x: x.shift(1).rolling(window=12, min_periods=1).sum())
    )

    gdf['wickets_last_12_balls'] = (
        gdf.groupby('match_id', group_keys=False)['is_wicket']
        .apply(lambda x: x.shift(1).rolling(window=12, min_periods=1).sum())
    )

    print("Merged data of matches and deliveries..")
    merged_df = gdf.merge(
        matches_df,
        how='left',
        left_on='match_id',
        right_on='id'
    )

    # Cleanup and optimize Data
    print("Merge data completed")

    for col in ['batting_team', 'bowling_team', 'team1', 'team2', 'toss_winner', 'winner']:
        merged_df[col] = merged_df[col].replace(team_map)

    cat_cols = [col for col in merged_df.columns if merged_df[col].dtype == object]
    for col in cat_cols:
        merged_df[col] = merged_df[col].str.strip().str.replace(r'\s+|\.|\,', '_', regex=True).str.lower()
    
    merged_df['city'] = merged_df['city'].fillna('')

    print("Completed data cleanup")

    # Derive fields
    merged_df['chase'] = (merged_df['winner'] == merged_df['batting_team']).astype(int)

    merged_df['current_score'] = merged_df.groupby('match_id')['total_runs'].cumsum()
    merged_df['wickets_down'] = merged_df.groupby('match_id')['is_wicket'].cumsum()
    merged_df['balls_remaining'] = (merged_df.target_overs * 6) - (merged_df.over * 6 + merged_df.ball)

    # Remove the match already won delivery rows from the dataset. 
    print(f"Current data size: {len(merged_df)}")
    merged_df = merged_df.drop(merged_df[merged_df.wickets_down == 10].index)
    merged_df = merged_df.drop(merged_df[merged_df.balls_remaining == 0].index)
    merged_df.reset_index(drop=True, inplace=True)
    print(f"Dropping win status data points. size: {len(merged_df)}")

    return merged_df

def derive_fields(mdf):
    mdf['required_run_rate'] = (mdf.target_runs - mdf.current_score) * 6 / mdf.balls_remaining
    return mdf


if __name__ == "__main__":


    low_cardinal_features = ['season', 'match_type', 'batting_team', 'toss_winner']
    high_cardinal_features = ['city']
    numeric_features = ['target_runs', 'current_score', 'wickets_down', 'balls_remaining', 'required_run_rate', 'runs_last_12_balls', 'wickets_last_12_balls' ]

    cols_of_interest = low_cardinal_features + high_cardinal_features + numeric_features + ['chase']


    mdf = prep_data()
    main_df = derive_fields(mdf)
    # Ensure the dataset is sorted by timeline.
    main_df = main_df.sort_values(by=['date', 'match_id', 'over', 'ball'])
    main_df = main_df[cols_of_interest]

    # Since it needs to be a time aware split, shuffle=False .. which results in past matches only for training and future matches for predictions.
    df_full_train, df_test = train_test_split(main_df, test_size=.2, shuffle=False, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=.25, shuffle=False, random_state=1)
    print(f"Length of df_train:{len(df_train)},  df_val:{len(df_val)},  df_test:{len(df_test)}")

    y_train = df_train.pop('chase')
    y_val = df_val.pop('chase')
    y_test = df_test.pop('chase')
    y_full_train = df_full_train.pop('chase')

    # Target Encoding for High Cardinality Features
    target_enc = ce.TargetEncoder(cols=high_cardinal_features)
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
            ('num', numerical_transformer, numeric_features),
            ('cat', categorical_transformer, low_cardinal_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )

    tscv = TimeSeriesSplit(n_splits=5)

    fold_aucs = []
    fold_accs = []
    model = LogisticRegression(
                C=0.01, max_iter=1000,
                random_state=21,
                class_weight='balanced'
            )
    for train_idx, val_idx in tscv.split(df_full_train):
        X_train_fold, X_val_fold = df_full_train.iloc[train_idx], df_full_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_full_train.iloc[train_idx], y_full_train.iloc[val_idx]

        target_enc = ce.TargetEncoder(cols=high_cardinal_features, smoothing=10)
        # Transform with target encoding and preprocessor
        X_train_enc = target_enc.fit_transform(X_train_fold, y_train_fold)
        X_val_enc   = target_enc.transform(X_val_fold)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train_enc, y_train_fold)

        y_pred_fold_acc = pipeline.predict(X_val_enc)
        y_pred_fold = pipeline.predict_proba(X_val_enc)[:, 1]
        fold_accs.append(accuracy_score(y_val_fold, y_pred_fold_acc))
        fold_aucs.append(roc_auc_score(y_val_fold, y_pred_fold))

    print(f"Trained Accuracy={np.mean(fold_accs):.4f}, ROC AUC={np.mean(fold_aucs):.4f}")


    X_test   = target_enc.transform(df_test)

    y_pred_acc = pipeline.predict(X_test)
    y_pred = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred_acc)
    auc = roc_auc_score(y_test, y_pred)
    print(f"Test Accuracy={acc:.3f}, ROC AUC={auc:.3f}")


    # Save Model

    with open(MODEL_FILE, "wb") as f_out:
        pickle.dump((target_enc, pipeline), f_out)

    print(f"Output enc, Model Pipeline saved to : {MODEL_FILE}")


