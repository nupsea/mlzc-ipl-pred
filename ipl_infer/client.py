import requests

match_state = {
    "season": "2025",
    "match_type": "league",
    "batting_team": "rcb",
    "toss_winner": "mi",
    "city": "bengaluru",
    "target_runs": 200,
    "current_score": 150,
    "wickets_down": 2,
    "balls_remaining": 25,
    "runs_last_12_balls": 20,
    "wickets_last_12_balls": 1,
}

print(" Accessing IPLPredictionService Model Endpoint ")

host = "localhost:9696"

res = requests.post(url=f"http://{host}/chase", json=match_state)

print(res.text)
