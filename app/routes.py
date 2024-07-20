from flask import request, jsonify
from app import app
import asyncio
import random
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

DB_FAISS_PATH = 'vectorstores7/db_faiss'

custom_prompt_template = """
Use the following pieces of information to answer the user's question in a comprehensive and informative way, leveraging the retrieved documents. If the retrieved documents are not helpful, try to provide relevant information or rephrase the question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=custom_prompt_template,
                         input_variables=['context', 'question'])

def retrieval_qa_chain(llm, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 5}),  # Retrieve top 5 documents
        return_source_documents=False,  # Do not return source documents
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    llm = ChatGroq(
        groq_api_key="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",  
        model_name="Llama3-8b-8192",
        max_tokens=8192,  
        temperature=0.5
    )
    return llm

async def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa = retrieval_qa_chain(llm, db)
    return qa

responses = [
    "Based on the information available, here is the relevant detail. If you need further clarification or assistance, please let me know:",
    "I have reviewed the documents related to your query. For any additional information or detailed support, feel free to ask:",
    "Here is the information extracted from the documents that should address your query. Should you require more specifics, please reach out:",
    "The following details are provided from the retrieved documents. If there's anything more specific you need or any other questions, I'm here to assist:",
    "I've gathered the pertinent details from the documents. For any further insights or additional queries, please let me know how I can assist further:"
]

@app.route('/chat', methods=['POST'])
async def chat():
    data = request.json
    question = data.get('question', '')
    
    chain = await qa_bot()
    res = await chain.acall(question)
    answer = res["result"] + "\n" + random.choice(responses)
    
    return jsonify({'answer': answer})

