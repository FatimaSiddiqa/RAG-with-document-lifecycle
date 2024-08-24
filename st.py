import requests
import os
import time
from tenacity import retry, stop_after_attempt, wait_fixed
import streamlit as st
import webbrowser

# Backend URL configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")

# Streamlit page configuration
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

# Page title and description
st.title("👽 Welcome to the Random Expert Chatbot!")
st.markdown("#### Your quirky assistant with expertise in world cuisines, famous authors, and the evolution of AI!")
st.markdown("I can also answer questions from documents you provide! 😸 However if any document is removed from my knowledge base, it'll take away my relevant knowledge too. 🥺")



if st.button("Learn more about the RAG model"):
    webbrowser.open_new_tab("http://localhost:8503")



# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# Retry decorator for backend requests
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def query_backend_with_retry(question):
    response = requests.post(f"{BACKEND_URL}/query", json={"question": question}, timeout=30)
    response.raise_for_status()
    return response.json()

def query_backend(question):
    try:
        st.session_state.processing = True
        return query_backend_with_retry(question)
    except requests.Timeout:
        return {"status": "error", "message": "Request timed out. The server might be overloaded. Please try again later."}
    except requests.RequestException as e:
        return {"status": "error", "message": f"Failed to connect to the backend: {str(e)}"}
    finally:
        st.session_state.processing = False

# Function to clear chat history
def clear_chat():
    st.session_state.messages = []
    st.experimental_rerun()

    
def display_evaluation_results(results):
    st.subheader("Evaluation Results")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            st.metric(label=metric, value=f"{value:.2f}")
        else:
            st.text(f"{metric}: {value}")

# Main content layout
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# Input prompt for the user
if prompt := st.chat_input("What do you wish to know?", disabled=st.session_state.processing):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with message_placeholder:
            st.markdown("Thinking... 🤔")
            st.spinner()

        result = query_backend(prompt)
        if result["status"] == "error":
            message_placeholder.error(result["message"])
        else:
            st.session_state.evaluation_results = result.get("feedback", {})
            
            if "answer" in result:
                full_response = result["answer"]    
            else:
                full_response = result.get("message", "An unknown error occurred.")
            
            message_placeholder.empty()
            message_placeholder.markdown(full_response)
        
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# This is where you put the Evaluate button and results display
if st.session_state.evaluation_results:
    if st.button("Evaluate"):
        display_evaluation_results(st.session_state.evaluation_results)

# Add Clear Chat button to the main screen
if st.button("Clear Chat"):
    clear_chat()

# Add a status indicator
if st.session_state.processing:
    st.warning("Backend is processing a request. Please wait...")


    
st.sidebar.markdown("---")
st.sidebar.markdown("The system monitors a folder for document changes and updates automatically.")
