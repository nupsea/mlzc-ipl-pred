import streamlit as st
import json
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import redis

# Set page configuration for better UI layout
st.set_page_config(page_title="T20 Live Prediction Dashboard", layout="wide")

st.title("T20 Live Prediction Dashboard")
st_autorefresh(interval=5000, limit=100, key="live_refresh")

# Default match ID 
MATCH_ID = "38238"
# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Allow user to input a different match ID
match_id_input = st.text_input("Enter Match ID:", value=MATCH_ID)

def get_redis_input():
    key = f"model-input-json:{match_id_input}"
    data = redis_client.get(key)
    if data:
        json_data = json.loads(data)
        fields_to_drop = {"match_id", "inning"}
        model_data = {k: v for k, v in json_data.items() if k not in fields_to_drop}
        return model_data
    else:
        st.error(f"No data found for key: {key}")
        return None

def get_prediction_from_rest(input_data):
    try:
        response = requests.post("http://localhost:9696/chase", json=input_data, timeout=2)
        return response.json()
    except Exception as e:
        st.error(f"Error calling prediction service: {e}")
        return None

# Cache the chart using st.cache_data so it doesn't rebuild unnecessarily.
@st.cache_data(show_spinner=False)
def get_chart(win_prob):
    labels = ["Chasing Team", "Defending Team"]
    values = [win_prob * 100, (1 - win_prob) * 100]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        transition={'duration': 500},
        height=500,
        width=500,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

# Create fixed columns with placeholders only once so that layout remains stable.
col1, col2 = st.columns([1, 2])
left_placeholder = col1.empty()
right_placeholder = col2.empty()

# Retrieve data from Redis
model_input = get_redis_input()

if model_input:
    # Create a score signature from key fields; adjust keys as per your data structure.
    score_signature = (
        model_input.get("target_runs"),
        model_input.get("balls_remaining"),
        model_input.get("current_score"),
        model_input.get("wickets_down")
    )
    
    # Check if the score has changed by comparing with the previous signature stored in session state.
    if "prev_score_signature" not in st.session_state or st.session_state.prev_score_signature != score_signature:
        # If there's a change (or first run), call the prediction service.
        prediction = get_prediction_from_rest(model_input)
        st.session_state.prev_score_signature = score_signature
        st.session_state.prev_prediction = prediction
    else:
        # No change; use the cached prediction.
        prediction = st.session_state.get("prev_prediction")

    if prediction:
        try:
            win_prob = float(prediction.get("win_probability", 0.0))
        except Exception as e:
            st.error(f"Error parsing win_probability: {e}")
            win_prob = 0.0
        
        chasing_team = prediction.get("chasing_team", "N/A")
        win_percentage_str = f"{win_prob * 100:.2f}%"
        
        # Left placeholder: Organized metrics display
        with left_placeholder.container():
            st.subheader("Match Statistics")
            # Display win probability with larger font size using HTML
            st.markdown(
                f"<p style='font-size:48px; margin:0;'>Win Probability: <b>{win_percentage_str}</b></p>",
                unsafe_allow_html=True,
            )
            st.write(f"**Chasing Team:** {chasing_team}")
            st.write("")  # Add a small break
            
            # First row: Three metrics
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Target Score", model_input.get("target_runs", "N/A"))
            col_b.metric("Balls Remaining", model_input.get("balls_remaining", "N/A"))
            col_c.metric("Current Score", model_input.get("current_score", "N/A"))
            
            # Second row: Three metrics, including new stats
            col_d, col_e, col_f = st.columns(3)
            col_d.metric("Wickets Down", model_input.get("wickets_down", "N/A"))
            col_e.metric("Runs Last 12 Balls", model_input.get("runs_last_12_balls", "N/A"))
            col_f.metric("Wickets Last 12 Balls", model_input.get("wickets_last_12_balls", "N/A"))
            
        # Right placeholder: Pie chart
        with right_placeholder.container():
            st.subheader("Win Probability Chart")
            # Use a fixed key for the chart to preserve its state between refreshes.
            st.plotly_chart(get_chart(win_prob), key="pie_chart", use_container_width=True)
    else:
        st.write("No prediction available yet.")
