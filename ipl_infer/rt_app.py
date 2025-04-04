import streamlit as st
import json
import requests
import matplotlib.pyplot as plt
from kafka import KafkaConsumer
from streamlit_autorefresh import st_autorefresh

st.write("Welcome to IPL T20 Live Prediction Dashboard")

# Auto-refresh every 10 seconds (10,000 milliseconds)
st_autorefresh(interval=10000, limit=100, key="live_refresh")

# Create and cache the Kafka consumer so it isn't recreated on every refresh
@st.cache_resource
def get_consumer():
    consumer = KafkaConsumer(
        't20-model-input',  # your topic for ball events
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',  # adjust as needed ('earliest' if you want all messages)
        consumer_timeout_ms=1000,      # stop waiting if no message arrives
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    return consumer

consumer = get_consumer()

def fetch_latest_ball_event():
    """Poll Kafka for the latest ball event message."""
    messages = consumer.poll(timeout_ms=1000)
    latest_event = None
    for topic_partition, msgs in messages.items():
        for msg in msgs:
            latest_event = msg.value
    return latest_event

# Poll Kafka for the latest ball event
ball_event = fetch_latest_ball_event()

if ball_event:
    st.write("Latest Ball Event:", ball_event)
    fields_to_drop = {"match_id", "inning"}
    model_input = {k: v for k, v in ball_event.items() if k not in fields_to_drop}

    # Send the ball event to the prediction service running on port 9696
    try:
        # Assume the prediction service expects a POST to /chase and returns JSON with a win_probability field
        response = requests.post("http://localhost:9696/chase", json=model_input, timeout=2)
        prediction_data = response.json()
        st.write("Prediction Data:", prediction_data)
        
        # Extract the win probability and generate a pie chart
        win_prob = float(prediction_data.get("win_probability", 0.0))
        labels = ["Chasing Team", "Defending Team"]
        sizes = [win_prob * 100, (1 - win_prob) * 100]
        explode = (0.1, 0)
        fig, ax = plt.subplots()
        ax.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error calling prediction service: {e}")
else:
    st.write("No new ball event available yet.")
