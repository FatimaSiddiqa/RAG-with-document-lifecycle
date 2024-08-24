import streamlit as st
import requests
import os
import time
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")
WATCH_DIR = r"C:\Users\Pc\Desktop\week5\rag"  
ARCHIVE_DIR = r"C:\Users\Pc\Desktop\week5"  

st.set_page_config(page_title="Document Management System", page_icon="📁", layout="wide")

st.title("Document Management System")

def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            status_info = response.json()
            return status_info.get("status") == "ok"
        return False
    except requests.RequestException:
        return False

def get_files_list():
    return [f for f in os.listdir(WATCH_DIR) if f.lower().endswith(('.pdf', '.docx', '.txt'))]

def upload_file(uploaded_file):
    if uploaded_file is not None:
        file_path = os.path.join(WATCH_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} uploaded successfully!")

def move_file(filename):
    source_path = os.path.join(WATCH_DIR, filename)
    destination_path = os.path.join(ARCHIVE_DIR, filename)
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        st.success(f"File {filename} moved to archive successfully!")
    else:
        st.error(f"File {filename} not found!")

class MyHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        st.rerun()

if 'files' not in st.session_state:
    st.session_state.files = get_files_list()

if check_backend_status():
    st.success("Backend is operational")
else:
    st.error("Backend is not responding. Please check the server.")

st.subheader("Upload a File")
uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt'])
if st.button("Upload"):
    upload_file(uploaded_file)
    st.session_state.files = get_files_list()

st.subheader("Archive a File")
file_to_move = st.selectbox("Select a file to archive", st.session_state.files)
if st.button("Archive"):
    move_file(file_to_move)
    st.session_state.files = get_files_list()

st.subheader("Current Files in Directory")
for file in st.session_state.files:
    st.text(file)

if st.button("Refresh File List"):
    st.session_state.files = get_files_list()
    st.rerun()

observer = Observer()
event_handler = MyHandler()
observer.schedule(event_handler, WATCH_DIR, recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()