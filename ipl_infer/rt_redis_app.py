import streamlit as st
import json
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import redis

st.write("Welcome to T20 Live Prediction Dashboard ")

# Auto-refresh every 10 seconds
st_autorefresh(interval=3000, limit=100, key="live_refresh")

MATCH_ID = "38238"
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

match_id_input = st.text_input("Enter Match ID:", value=MATCH_ID)

def get_redis_input():
    key = f"model-input-json:{MATCH_ID}"
    data = redis_client.get(key)


    if data:
        json_data = json.loads(data)
        fields_to_drop = {"match_id", "inning"}
        model_data = {k: v for k, v in json_data.items() if k not in fields_to_drop}
        return model_data
    else:
        st.error(f"No data found for key: {key}")
        return None
    

# Function to get the latest prediction from REST service
def get_prediction_from_rest(input_data):
    try:
        response = requests.post("http://localhost:9696/chase", json=input_data, timeout=2)
        return response.json()
    except Exception as e:
        st.error(f"Error calling prediction service: {e}")
        return None

# Retrieve prediction data
model_input = get_redis_input()
prediction = get_prediction_from_rest(model_input)

if prediction:
    try:
        win_prob = float(prediction.get("win_probability", 0.0))
    except Exception as e:
        st.error(f"Error parsing win_probability: {e}")
        win_prob = 0.0
    labels = ["Chasing Team", "Defending Team"]
    values = [win_prob * 100, (1 - win_prob) * 100]

    # Use Plotly to create a pie chart with a smooth transition
    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.3)]
    )
    # Update layout to animate changes (transition duration in ms)
    fig.update_layout(transition={'duration': 500})
    
    st.plotly_chart(fig, use_container_width=True)
    st.write("Latest Prediction:", prediction)
else:
    st.write("No prediction available yet.")
