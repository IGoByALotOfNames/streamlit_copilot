import os
import time
import streamlit as st

def watch_directory(path_to_watch):
    before = dict([(f, None) for f in os.listdir(path_to_watch)])
    while True:
        time.sleep(10)  # Pause for 10 seconds
        after = dict([(f, None) for f in os.listdir(path_to_watch)])
        added_files = [f for f in after if not f in before]
        if added_files:
            for file in added_files:
                st.write(f"New file added: {file}")
        before = after

# Example usage
watch_directory('')
