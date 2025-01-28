import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import streamlit as st
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Initialize sentence-transformers for embeddings (since Gemini doesn't have an embeddings API)
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# configure PostgreSQL connection
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cursor = conn.cursor()

# sample document chunks
documents=[
    "Artificial Intelligence is a field of computer science that focuses on creating intelligent machines that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language understanding.",
    "Machine Learning is a subset of Artificial Intelligence that involves training algorithms to make predictions or decisions based on data, without being explicitly programmed.",
    "Natural Language Processing (NLP) is a field of study that focuses on the interaction between computers and humans using natural language.",
    "Deep Learning is a subset of Machine Learning that involves training neural networks with multiple layers to learn representations of data.",
    "Reinforcement Learning is a type of Machine Learning that involves training an agent to make decisions in an environment by giving it rewards or punishments for its actions.",
    "Computer Vision is a field of study that focuses on enabling computers to understand and interpret visual information from the world around them.",
    "Robotics is a field of study that focuses on creating machines that can perform tasks autonomously or with minimal human supervision.",
]

st.title("Interactive Q&A System with Retrieval Augmented Generation")
st.write("This application demonstrates the steps in a Q&A system with Retrieval Augmented Generation (RAG).")
### Step 1: Document Chunking
st.header("Step 1: Document Chunking")
st.write("Here are the document chunks that will be used for this document")
for i, doc in enumerate(documents,start=1):
    st.write(f"**Chunk {i}:** {doc}")

# Generate and Store Embeddings in PostgreSQL with User Display
embeddings=[]


# Step 2: Generate and Display Embeddings
st.header("Step 2: Generating Embeddings")
st.write("Each Document Chunk is converted into an embedding vector representation.")

def get_embedding(text):
    return embedding_model.encode(text).tolist()

for doc in documents:
    # Generate Embedding
    embedding = get_embedding(doc)
    embeddings.append(embedding)
    
    #Insert the document Text and Embedding into the Table
    cursor.execute("INSERT INTO document_chunks (content,embedding) VALUES (%s, %s)", (doc,embedding))
    st.write(f"Embedding for Chunks '{doc}':")
    st.write(embedding)

conn.commit()

### Step 3: Retrieving Relevant Chunks for a Question
st.header("Step 3: Retrieving Relevant Chunks for a Question")
question=st.text_input("Enter your question:")

def get_relevant_chunks(question, top_k=3):
    # Generate Embedding for the Question
    question_embedding = get_embedding(question)
    cursor.execute("""
        SELECT content
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """, (question_embedding, top_k))
    relevant_chunks = [row[0] for row in cursor.fetchall()] 
    return relevant_chunks

if question:
    st.write(f"**Embedding for the questions** '{question}' :")
    question_embedding = get_embedding(question)
    st.write(question_embedding)
    relevant_chunks = get_relevant_chunks(question)
    st.write(f"Top relevant chunks retrieved:")
    for i, chunks in enumerate(relevant_chunks, start=1):
        st.write(f"{i}. {chunks}")

    ###  Step 4: Generate and display the answer 
    st.header("Step 4: Generate answer using Gemini")
    context = "\n".join(f"{i+1}. {chunks}" for i, chunks in enumerate(relevant_chunks))
    prompt = f"""Using the following information:
{context}

Answer the question: {question}

Provide a clear and concise answer based only on the information provided above."""

    # Generate response using Gemini
    response = model.generate_content(prompt)
    answer = response.text.strip()
    st.write("**Generated Answer:**")
    st.write(answer)

# closing the connections
cursor.close()
conn.close()