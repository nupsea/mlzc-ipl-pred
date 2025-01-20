import pickle
import os
import pandas as pd


def load():
    version = os.getenv("MODEL_VERSION", "v1")
    model_dir = os.getenv("MODEL_DIR", "../MODEL")

    MODEL_FILE = os.path.join(model_dir, f"ipl_chase_pred_{version}.bin")

    with open(MODEL_FILE, "rb") as m_fh:
        target_enc, pipeline = pickle.load(m_fh)

    return target_enc, pipeline


def derive_fields(mdf):
    mdf["required_run_rate"] = (
        (mdf.target_runs - mdf.current_score) * 6 / mdf.balls_remaining
    )
    return mdf


def validate_inputs(data):
    """
    Raise ValueError if any input field is unrealistic.
    """
    # 1) Score cannot be negative
    if data["current_score"] < 0:
        raise ValueError("current_score cannot be negative.")

    # 2) Wickets must be between 0 and 10
    if data["wickets_down"] < 0 or data["wickets_down"] > 10:
        raise ValueError("wickets_down must be between 0 and 10.")

    # 3) Balls remaining can't be negative; if T20, max 120 total balls in an innings
    if data["balls_remaining"] < 0:
        raise ValueError("balls_remaining cannot be negative.")

    # If you assume standard T20 and no DLS changes, you might also check an upper bound:
    if data["balls_remaining"] > 120:
        raise ValueError("balls_remaining cannot exceed 120 for a standard T20 inning.")

    # 4) Target runs must be at least > 0
    if data["target_runs"] <= 0:
        raise ValueError("target_runs must be greater than 0.")

    # 5) Wickets in last 12 balls can't be negative, etc.
    if data["wickets_last_12_balls"] < 0 or data["wickets_last_12_balls"] > 9:
        raise ValueError("wickets_last_12_balls must be b/w 0 and 9.")

    # 6) Runs in last 12 balls can't be negative, etc.
    if data["runs_last_12_balls"] < 0:
        raise ValueError("runs_last_12_balls cannot be negative.")

    # 7) Exit if a win detected
    if data["current_score"] >= data["target_runs"]:
        print(f"{data['batting_team']} has won already :) ")
        exit()
    elif data["wickets_down"] == 10:
        print(f"{data['batting_team']} has lost already :( ")
        exit()


def predict(input):

    validate_inputs(input)

    target_enc, pipeline = load()
    df_t = pd.DataFrame([input])
    df_t = derive_fields(df_t)
    X = target_enc.transform(df_t)
    y_pred = pipeline.predict_proba(X)[:, 1][0]

    print(
        f"Chasing team '{str(input['batting_team']).upper()}' has a Predicted Win probability of: {y_pred * 100:.2f}% "
    )


if __name__ == "__main__":
    c = {
        "season": "2025",
        "match_type": "league",
        "batting_team": "gt",
        "toss_winner": "mi",
        "city": "mumbai",
        "target_runs": 200,
        "current_score": 180,
        "wickets_down": 5,
        "balls_remaining": 24,
        "runs_last_12_balls": 10,
        "wickets_last_12_balls": 2,
    }
    predict(c)
