import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
BACKEND_URL = "http://localhost:8000"

st.title("RAG Evaluation App")

@st.cache_data(ttl=3600)  
def evaluate_rag():
    with st.spinner("Evaluating RAG model... This may take a few minutes."):
        time.sleep(5)  
        response = requests.post(f"{BACKEND_URL}/evaluate_rag")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to evaluate RAG. Please check the backend.")
            return None

def evaluate_rag():
    response = requests.post(f"{BACKEND_URL}/evaluate_rag")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to evaluate RAG. Please check the backend.")
        return None

results = evaluate_rag()

if results:
    st.subheader("Evaluation Results")
    
    df = pd.DataFrame([results])
    st.table(df)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(results.keys()),
            y=list(results.values()),
            text=[f"{value:.2f}" for value in results.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="RAG Evaluation Metrics",
        xaxis_title="Metrics",
        yaxis_title="Scores",
        yaxis=dict(range=[0, 1]) 
    )
    
    st.plotly_chart(fig)

st.markdown("---")
st.markdown("""
This app evaluates a Retrieval-Augmented Generation (RAG) model using the following metrics:
- Context Precision: Measures how precise the retrieved contexts are.
- Faithfulness: Measures how faithful the generated answer is to the given context.
- Answer Relevancy: Measures how relevant the generated answer is to the question.
- Context Recall: Measures how well the model recalls relevant information from the context.
""")

if st.button("Refresh Evaluation"):
    st.cache_data.clear()
    st.experimental_rerun()

if st.button("Back to Chat"):
    st.markdown('<script>window.close();</script>', unsafe_allow_html=True)
