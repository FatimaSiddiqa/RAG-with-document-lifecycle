from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import os
import ast
import json
import pandas as pd
import datasets
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
from datasets import Dataset
import math

def handle_nan(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


app = FastAPI()

# Set up environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
if not os.environ["GROQ_API_KEY"]:
    raise ValueError("GROQ_API_KEY is not set in environment variables")

# Load the LLM
llm = ChatGroq(model="llama3-8b-8192")

# Define the prompt template
prompt_template = """
You are an expert in World Cuisines, AI evolution, famous authors, and Emumba.
Respond to the question based on the provided context information, without relying on prior knowledge.
If the context doesn't contain relevant information, say "I don't have enough information to answer that."
Always cite the specific parts of the context in your answer.
Always justify your answer by supporting it using the relevant context.
Answer the question confidently.
Don't say 'according to the context', 'based on the context' or similar.
Do not give irrelevant information.

CONTEXT INFORMATION:
{context}

QUESTION: {question}

ANSWER: Provide a detailed answer, citing specific parts of the context.
CITATIONS: Please include citations in the format [1], [2], etc. to reference the context.
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(prompt_template)

# Define Pydantic models for request and response data
class QuestionAnswer(BaseModel):
    question: str
    ground_truth: str
    contexts: str

class RagasResult(BaseModel):
    context_precision: float | None
    faithfulness: float | None
    answer_relevancy: float | None
    context_recall: float | None

# Initialize global variables for vectorstore and retriever
vectorstore = None
retriever = None

# File to store evaluation results
RESULTS_FILE = "evaluation_results.json"

# Function to load results from file
def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)
            return {k: handle_nan(v) for k, v in data.items()}
    return None

# Function to save results to file
def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f)

# Load results when the app starts
evaluation_results = load_results()

# Function to load and process PDFs
def load_and_process_pdfs(pdf_folder_path: str):
    documents = []
    for file in os.listdir(pdf_folder_path):
        file_path = os.path.join(pdf_folder_path, file)
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
    semantic_chunker = SemanticChunker(HuggingFaceEmbeddings())
    splits = semantic_chunker.create_documents([d.page_content for d in documents])
    return splits, documents

# Function to initialize the vectorstore
def initialize_vectorstore(splits):
    return FAISS.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())

def parse_contexts(contexts_str):
    try:
        return ast.literal_eval(contexts_str)
    except:
        try:
            return json.loads(contexts_str)
        except:
            return []

@app.post("/load_pdfs") 
async def load_pdfs(): 
    global vectorstore, retriever
    pdf_folder_path = r"C:\Users\Pc\Desktop\week5\rag" 
    splits, documents = load_and_process_pdfs(pdf_folder_path) 
    vectorstore = initialize_vectorstore(splits) 
    return {"message": "PDFs loaded and vectorstore initialized successfully."}

def perform_evaluation():
    global vectorstore, retriever, evaluation_results
    pdf_folder_path = r"C:\Users\Pc\Desktop\week5\rag" 
    splits, documents = load_and_process_pdfs(pdf_folder_path) 
    vectorstore = initialize_vectorstore(splits) 
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    qa = pd.read_csv(r"C:\Users\Pc\Downloads\testset.csv")
    qa['contexts'] = qa['contexts'].apply(parse_contexts)
    
    features = datasets.Features(
        {
            "question": datasets.Value("string"),
            "contexts": datasets.Sequence(datasets.Value("string")),
            "ground_truth": datasets.Value("string"),
            "evolution_type": datasets.Value("string"),
            "metadata": datasets.Value("string"),
            "episode_done": datasets.Value("bool"),
        }
    )

    dataset = datasets.Dataset.from_pandas(qa, features=features)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    langchain_llm = llm
    langchain_embeddings = HuggingFaceEmbeddings()
            
    ragas_data = []
    for row in dataset:
        question = row['question']
        ground_truth = row['ground_truth']
        contexts = row['contexts']
        rag_answer = rag_chain.invoke(question)

        ragas_data.append({
            "question": question,
            "answer": rag_answer,
            "contexts": contexts,
            "ground_truths": [ground_truth]
        })

    ragas_dataset = Dataset.from_list(ragas_data)

    result = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
        raise_exceptions=False,
        llm=langchain_llm,
        embeddings=langchain_embeddings,
    )

    evaluation_results = {
        "context_precision": handle_nan(result['context_precision']),
        "faithfulness": handle_nan(result['faithfulness']),
        "answer_relevancy": handle_nan(result['answer_relevancy']),
        "context_recall": handle_nan(result['context_recall'])
    }
    save_results(evaluation_results)

@app.post("/evaluate_rag", response_model=RagasResult)
async def evaluate_rag():
    global evaluation_results
    
    if evaluation_results is None:
        perform_evaluation()
    
    return evaluation_results

@app.post("/reset_evaluation")
async def reset_evaluation():
    global evaluation_results
    evaluation_results = None
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    return {"message": "Evaluation results have been reset. Next call to /evaluate_rag will perform a new evaluation."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)