# app/streamlit_app.py
import streamlit as st
import json
from pathlib import Path
import time

st.title("Real-Time Anomaly Detection")
file = Path("data/current_anomaly.json")

while True:
    if file.exists():
        a = json.loads(file.read_text())
        st.subheader(f"Room {a['resourceid']} @ {a['timestamp']}")
        st.write(f"Error: {a['error']} > {a['threshold']}")
        col1, col2 = st.columns(2)
        if col1.button("False Alarm"):
            file.unlink()
            st.success("Logged: False")
        if col2.button("True Anomaly"):
            file.unlink()
            st.success("Logged: True")
    else:
        st.info("Monitoring...")
    time.sleep(1)
    st.rerun()
