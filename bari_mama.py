

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import requests
chat_memory = {}

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load and prepare documents once at startup
loader = TextLoader('Bari_Mama_Complete_Document.txt', encoding='utf-8')
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Chroma.from_documents(chunks, embedding)

# FastAPI app
app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    query: str
    session_id: str
 

@app.post("/chat")
def chat_endpoint(payload: QueryRequest):
    query = payload.query
    session_id = payload.session_id

    # Initialize memory if not present
    if session_id not in chat_memory:
        chat_memory[session_id] = []

    # Get chat history for that session
    history = chat_memory[session_id]

    # Search context from vectorstore
    similar = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in similar])

    # Prepare previous conversation memory
    conversation_log = ""
    for user_msg, assistant_msg in history[-5:]:  # limit to last 5 for token safety
        conversation_log += f"Customer: {user_msg}\nAssistant: {assistant_msg}\n"

    # Create full prompt
    prompt = f"""You are BARI MAMMA ASSISTANT.  
Your job is to reply based on the context.  
Keep replies short (under 90 words), friendly, and in simple English.
If Customer Interested in Buying then you can ask to show some images and for images just return **URL** avaliable in context.

If Customer make ORDER then no need to look in context  just thanks in a good way.

Be professional and dont say **Hi**, **Hello** again and again.


Context:  
{context}

Conversation History:  
{conversation_log}

Query:  
Customer: {query}  
Assistant:
"""


    # Call Groq API
    api_key = "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"  # Replace with your key
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 150
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    reply = result['choices'][0]['message']['content']

    # Save the new interaction
    chat_memory[session_id].append((query, reply))

    return {"reply": reply}

