from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = Flask(__name__)

retrieval_chain = None

def load_and_split_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)
    return documents

def create_faiss_db(documents):
    embeddings = OllamaEmbeddings(model='llama3')
    db = FAISS.from_documents(documents, embedding=embeddings)
    return db

def create_retrieval_chain_with_ollama(db):
    llm = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_template("""
    Handle greetings and goodbyes and small talk with the user without providing any useful information.
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    Give the answer precisely without missing any important details.
    Act like you are directly answering the user's question without mentioning words like "Based on the context".
    when clubs is reffered treat it as student clubs and give all the details about it.
    <context>
    {context}
    </context>
    Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

@app.route('/init', methods=['POST'])
def initialize():
    global retrieval_chain
    pdf_path = request.json.get('pdf_path')
    # documents = load_and_split_documents(pdf_path)
    documents = load_and_split_documents("Chatbot.pdf")
    db = create_faiss_db(documents)
    retrieval_chain = create_retrieval_chain_with_ollama(db)
    return jsonify({"message": "Chatbot initialized with PDF"})

@app.route('/chat', methods=['POST'])
def chat():
    global retrieval_chain
    if retrieval_chain is None:
        return jsonify({"error": "Chatbot not initialized. Please upload a PDF first."})
    
    user_input = request.json.get('input')
    if user_input.lower() == "bye":
        return jsonify({"response": "Goodbye!"})
    
    response = retrieval_chain.invoke({"input": user_input})
    return jsonify({"response": response['answer']})

if __name__ == "__main__":
    app.run(debug=True)