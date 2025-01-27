
import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from predict import predict

st.write("Welcome to IPL T20 second-half for predicting chasing outcome.")


# Get inputs
season = st.text_input("Season:", key="season", placeholder="2025")

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

match_type = st.selectbox(
    'Match Type?',
     ('league', 'semi-final', 'final'))

chasing_team_full = st.selectbox(
    'Chasing team',
    list(team_map.keys())
)
chasing_team = team_map.get(chasing_team_full).lower()

toss_won_team_full = st.selectbox(
    'Toss Winning team',
    list(team_map.keys())
)
toss_won_team = team_map.get(toss_won_team_full).lower()

city = st.text_input("City:", key="city")
target_runs = st.number_input("Target Runs:", key="target_runs", step=1)
current_score = st.number_input("Current Score:", key="current_score", step=1)
wickets_down = st.number_input("Wickets down:", key="wickets_down",step=1)
balls_remaining = st.number_input("Balls remaining:", key="balls_remaining",step=1)
runs_last_12_balls = st.number_input("Runs scored last 2 overs:", key="runs_last_12_balls",step=1)
wickets_last_12_balls = st.number_input("Wickets lost last 2 overs:", key="wickets_last_12_balls",step=1)

match_state = {
    "season": season,
    "match_type": match_type,
    "batting_team": chasing_team,
    "toss_winner": toss_won_team,
    "city": city,
    "target_runs": target_runs,
    "current_score": current_score,
    "wickets_down": wickets_down,
    "balls_remaining": int(st.session_state.balls_remaining),
    "runs_last_12_balls": runs_last_12_balls,
    "wickets_last_12_balls": wickets_last_12_balls
}


def validate_inputs(data) -> bool:
    errors = []
    
    if data["current_score"] < 0:
        errors.append("current_score cannot be negative.")
    if not (0 <= data["wickets_down"] <= 10):
        errors.append("wickets_down must be between 0 and 10.")
    if data["balls_remaining"] < 0 or data["balls_remaining"] > 120:
        errors.append("balls_remaining must be in [0, 120].")
    if data["target_runs"] <= 0:
        errors.append("target_runs must be > 0.")
    if data["wickets_last_12_balls"] < 0 or data["wickets_last_12_balls"] > 9:
        errors.append("wickets_last_12_balls must be in [0, 9].")
    if data["wickets_last_12_balls"] > data["wickets_down"]:
        errors.append("wickets_last_12_balls cannot exceed wickets_down.")
    if data["runs_last_12_balls"] < 0:
        errors.append("runs_last_12_balls cannot be negative.")
    if data["runs_last_12_balls"] > data["current_score"]:
        errors.append("runs_last_12_balls cannot exceed current_score.")
    if data["current_score"] >= data["target_runs"]:
        errors.append(f"{data['batting_team']} has already won!")
    elif data["wickets_down"] == 10:
        errors.append(f"{data['batting_team']} has already lost (all out)!")

    if errors:
        for e in errors:
            st.error(e)
        return False
    return True

valid = validate_inputs(match_state)

# st.write("Current match state:", match_state)

predict_clicked = st.button("Predict", disabled=not valid)
if predict_clicked:

    host = "localhost:9696"
    res = requests.post(
        url=f'http://{host}/chase',
        json=match_state
    )

    response = res.json()
    st.write(response)


    prob = float(response["win_probability"])

    labels = ['Chasing Team', 'Defending Team']
    sizes = [prob * 100, (1 - prob) * 100] 
    explode = (0.1, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')

    st.pyplot(fig1)