import traceback
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel
from collections import deque
from transformers import pipeline
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import threading
import logging
import torch
from trulens_eval import TruChain, Tru
from trulens_eval.feedback.provider.langchain import Langchain
from trulens_eval import Feedback
import numpy as np
from trulens_eval.app import App

exit_flag = False

update_in_progress=False
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
if not os.environ["GROQ_API_KEY"]:
    raise ValueError("GROQ_API_KEY is not set in environment variables")

WATCH_DIR = r"./rag"
if not os.path.exists(WATCH_DIR):
    raise FileNotFoundError(f"WATCH_DIR does not exist: {WATCH_DIR}")

vectorstore = None
rag_chain = None
update_queue = deque()
document_cache = {}
retriever = None
llm = ChatGroq(model="llama3-8b-8192")
system_status = "ready"



def load_single_document(file_path):
    logger.debug(f"Loading document: {file_path}")
    try:
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return None

        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = file_path
        return docs
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    

def process_documents(documents):
    if not documents:
        logger.warning("No documents to process")
        return []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model_kwargs = {'device': device}
    encode_kwargs = {'device': device, 'batch_size': 32}
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        logger.warning(f"Error initializing HuggingFaceEmbeddings: {e}")
        logger.warning("Falling back to default settings")
        embeddings = HuggingFaceEmbeddings()
    
    semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])

    if not semantic_chunks:
        logger.warning("No semantic chunks created")
        return []

    for i, chunk in enumerate(semantic_chunks):
        if i < len(documents):
            chunk.metadata['source'] = documents[i].metadata['source']
        else:
            chunk.metadata['source'] = documents[-1].metadata['source']

    return semantic_chunks

def load_and_process(pdf_folder_path):
    file_paths = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) 
                  if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    
    documents = []
    for file_path in tqdm(file_paths, desc="Loading documents"):
        try:
            docs = load_single_document(file_path)
            if docs:
                documents.extend(docs)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            logger.error(traceback.format_exc())

    processed_documents = process_documents(documents)
    return processed_documents

def update_document(file_path, vectorstore):
    global system_status
    logger.debug(f"Updating document: {file_path}")
    system_status = "updating"
    documents = load_single_document(file_path)
    if documents:
        chunks = process_documents(documents)
        
        existing_ids = [id for id, doc in vectorstore.docstore._dict.items() 
                        if doc.metadata.get('source') == file_path]
        
        if existing_ids:
            vectorstore.delete(existing_ids)
        
        vectorstore.add_documents(chunks)
        document_cache[file_path] = chunks
        logger.info(f"Document updated: {file_path}")
    else:
        logger.warning(f"No documents loaded from {file_path}")
    system_status = "ready"
    logger.info("Vectorstore indexes updated. Ready for new questions.")

def delete_document(file_path, vectorstore):
    global system_status
    logger.debug(f"Deleting document: {file_path}")
    system_status = "updating"
    ids_to_delete = [id for id, doc in vectorstore.docstore._dict.items() 
                     if doc.metadata.get('source') == file_path]
    if ids_to_delete:
        vectorstore.delete(ids_to_delete)
        document_cache.pop(file_path, None)
        logger.info(f"Document deleted: {file_path}")
    else:
        logger.warning(f"No document found to delete: {file_path}")
    system_status = "ready"
    logger.info("Vectorstore indexes updated. Ready for new questions.")



def initialize_vectorstore(splits):
    logger.debug("Initializing vectorstore")
    for split in splits:
        split.metadata['source'] = split.metadata.get('source', 'unknown')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device for vectorstore: {device}")
    
    model_kwargs = {'device': device}
    encode_kwargs = {'device': device, 'batch_size': 32}
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return FAISS.from_documents(documents=splits, embedding=embeddings)
    except Exception as e:
        logger.error(f"Error initializing vectorstore: {e}")
        logger.error(traceback.format_exc())
        raise

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chunk_if_needed(text, max_tokens=6000):
    num_tokens = num_tokens_from_string(text)
    if num_tokens > max_tokens:
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=100)
        chunks = splitter.split_text(text)
        return chunks[:3] 
    return [text]



def inference():
    global retriever, vectorstore

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    prompt_template = """
    You are an expert in World Cuisines, AI evolution,and famous authors.
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

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.3)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


    return rag_chain

def get_trulens_feedback(rag_chain, question):
    try:
        def evaluate_trulens():
            tru = Tru()
            tru.reset_database()

            provider = Langchain(chain=llm)
            context = App.select_context(rag_chain)

            f_groundedness = (
                Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
                .on(context.collect())  # collect context chunks into a list
                .on_output()
            )

            f_answer_relevance = (
                Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
                .on_input_output()
            )

            f_context_relevance = (
                Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
                .on_input()
                .on(context)
                .aggregate(np.mean)
            )

            tru_recorder = TruChain(rag_chain, app_id='ragchain', feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness])

            with tru_recorder as recording:
                llm_response = rag_chain.invoke(question)
            last_record = recording.records[-1]

            feedback_results = {}
            for feedback, feedback_result in last_record.wait_for_feedback_results().items():
                    feedback_results[feedback.name] = feedback_result.result

            # Extract citations from context
            citations = []
            for chunk in context:
                if isinstance(chunk, (str, list, tuple)):
                    elements = [chunk] if isinstance(chunk, str) else chunk
                    for element in elements:
                        if isinstance(element, str):
                            for sentence in element.split("."):
                                if sentence.strip():
                                    citations.append(f"[{len(citations)+1}] {sentence.strip()}")

            # Include citations in response
            response = llm_response + "\n\nCitations:\n" + "\n".join(citations)

            return { "feedback": feedback_results}

        with ThreadPoolExecutor() as executor:
                future = executor.submit(evaluate_trulens)
                return future.result()

    except Exception as e:
        logger.exception(f"Error in get_trulens_feedback: {e}")
        return {"error": str(e)}

    


def get_last_modified_dict(path: str) -> dict:
    data = os.walk(path)
    return_data = dict()
    for root, _, files in data:
        for file in files:
            if file.lower().endswith(('.pdf', '.docx', '.txt')):
                file_path = os.path.join(root, file)
                return_data[file_path] = os.stat(file_path).st_mtime
    return return_data

def get_unique_data(old_dict: dict, new_dict: dict) -> dict:
    return_data = {
        "changed": [],
        "new": [],
        "deleted": []
    }
    if old_dict != new_dict:
        return_data["changed"] = [
            file for file in set(old_dict.keys()).intersection(set(new_dict.keys()))
            if old_dict[file] != new_dict[file]
        ]
        return_data["new"] = list(set(new_dict.keys()).difference(set(old_dict.keys())))
        return_data["deleted"] = list(set(old_dict.keys()).difference(set(new_dict.keys())))

    return return_data

def update_thread_function():
    global update_in_progress, vectorstore, rag_chain, exit_flag
    logger.info("Update thread started")
    current_data = get_last_modified_dict(WATCH_DIR)

    while not exit_flag:
        try:
            new_data = get_last_modified_dict(WATCH_DIR)
            if new_data != current_data:
                update_in_progress = True
                logger.info("Changes detected. Updating...")
                
                unique_data = get_unique_data(current_data, new_data)

                if unique_data["changed"]:
                    logger.info(f"Changed files: {', '.join(unique_data['changed'])}")
                if unique_data["new"]:
                    logger.info(f"New files: {', '.join(unique_data['new'])}")
                if unique_data["deleted"]:
                    logger.info(f"Deleted files: {', '.join(unique_data['deleted'])}")

                current_data = new_data

                logger.info("Reloading and processing documents")
                splits = load_and_process(WATCH_DIR) 
                logger.info("Reinitializing vector store")
                vectorstore = initialize_vectorstore(splits)
                logger.info("Updating inference chain")
                rag_chain = inference()
                
                update_in_progress = False
                logger.info("Update complete. Ready for new questions.")

            time.sleep(1) 

        except Exception as e:
            logger.exception(f"Error in update thread: {e}")
            update_in_progress = False

    logger.info("Update thread exiting")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, rerank, rag_chain
    
    try:
        logger.info("Starting application initialization")
        logger.info("Loading and processing documents")
        splits = load_and_process(WATCH_DIR)
        logger.info("Initializing vectorstore")
        vectorstore = initialize_vectorstore(splits)
        logger.info("Setting up inference chain")
        rag_chain = inference()

        logger.info("Starting update thread")
        update_thread = threading.Thread(target=update_thread_function)
        update_thread.daemon = True
        update_thread.start()
        
        logger.info("Application startup complete")
        yield
    except Exception as e:
        logger.exception(f"Error during startup: {e}")
        raise
    finally:
        logger.info("Shutting down application")
        global exit_flag
        exit_flag = True
        if 'update_thread' in locals():
            update_thread.join(timeout=5)
        logger.info("Shutdown complete")

app = FastAPI(lifespan=lifespan)

class Question(BaseModel):
    question: str

@app.post("/query")
async def query(question: Question):
    global rag_chain, update_in_progress

    if update_in_progress:
        return {"status": "updating", "message": "System is currently updating. Please try again in a moment."}

    if rag_chain is None:
        return {"status": "initializing", "message": "System not initialized. Please wait a moment and try again."}

    try:
        logger.info(f"Invoking rag_chain with question: {question.question}")
        response = rag_chain.invoke(question.question)
        logger.info("rag_chain invocation successful")
        
        logger.info("Getting TruLens feedback")
        feedback_results = get_trulens_feedback(rag_chain, question.question)
        logger.info("TruLens feedback retrieved successfully")
        
        return {"status": "success", "answer": response, "feedback": feedback_results}
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        return {"status": "error", "message": f"An error occurred while processing your query: {str(e)}"}
    


@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
