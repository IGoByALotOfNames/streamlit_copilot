import os
import time
import streamlit as st

def watch_directory():
    before = dict([(f, None) for f in os.listdir()])
    while True:
        time.sleep(10)  # Pause for 10 seconds
        after = dict([(f, None) for f in os.listdir()])
        added_files = [f for f in after if not f in before]
        if added_files:
            for file in added_files:
                st.write(f"New file added: {file}")
        before = after
st.write("huh")
# Example usage
watch_directory()
