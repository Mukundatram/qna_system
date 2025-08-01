import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import streamlit as st
import psycopg2
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure Gemini using Streamlit Secrets
genai.configure(api_key=st.secrets["google_api_key"])
model = genai.GenerativeModel('gemini-2.5-flash')

# Initialize sentence-transformers with smaller model for deployment
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
assert len(embedding_model.encode("test")) == 384

# Configure PostgreSQL connection for Neon
conn = psycopg2.connect(
    host=st.secrets["postgres"]["host"],
    port=st.secrets["postgres"]["port"],
    dbname=st.secrets["postgres"]["dbname"],
    user=st.secrets["postgres"]["user"],
    password=st.secrets["postgres"]["password"],
    sslmode='require'
)

cursor = conn.cursor()

# Database setup with error handling
try:
    # Create vector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    
    # Check if table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'document_chunks'
        )
    """)
    table_exists = cursor.fetchone()[0]

    # Create table only if it doesn't exist
    if not table_exists:
        cursor.execute("""
            CREATE TABLE document_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(384)
            )
        """)
        conn.commit()

except Exception as e:
    st.error(f"Database setup error: {str(e)}")
    conn.rollback()

# Sample document chunks
documents = [
    "Artificial Intelligence is a field of computer science that focuses on creating intelligent machines that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language understanding.",
    "Machine Learning is a subset of Artificial Intelligence that involves training algorithms to make predictions or decisions based on data, without being explicitly programmed.",
    "Natural Language Processing (NLP) is a field of study that focuses on the interaction between computers and humans using natural language.",
    "Deep Learning is a subset of Machine Learning that involves training neural networks with multiple layers to learn representations of data.",
    "Reinforcement Learning is a type of Machine Learning that involves training an agent to make decisions in an environment by giving it rewards or punishments for its actions.",
    "Computer Vision is a field of study that focuses on enabling computers to understand and interpret visual information from the world around them.",
    "Robotics is a field of study that focuses on creating machines that can perform tasks autonomously or with minimal human supervision.",
]

# UI Components
st.title("Interactive Q&A System with Retrieval Augmented Generation")
st.write("This application demonstrates the steps in a Q&A system with Retrieval Augmented Generation (RAG).")

# Step 1: Document Chunking
st.header("Step 1: Document Chunking")
st.write("Here are the document chunks that will be used for this document")
for i, doc in enumerate(documents, start=1):
    st.write(f"**Chunk {i}:** {doc}")

# Step 2: Generate and Store Embeddings
st.header("Step 2: Generating Embeddings")
st.write("Each Document Chunk is converted into an embedding vector representation.")

def get_embedding(text):
    return embedding_model.encode(text).tolist()

try:
    # Insert documents
    for doc in documents:
        embedding = get_embedding(doc)
        cursor.execute(
            "INSERT INTO document_chunks (content, embedding) VALUES (%s, %s)",
            (doc, embedding)
        )
        st.write(f"Embedding for Chunk '{doc[:50]}...':")
        st.write(embedding[:5] + ["..."])  # Show partial embedding
    
    conn.commit()

except Exception as e:
    st.error(f"Embedding insertion error: {str(e)}")
    conn.rollback()

# Step 3: Retrieving Relevant Chunks
st.header("Step 3: Retrieving Relevant Chunks for a Question")
question = st.text_input("Enter your question:")

def get_relevant_chunks(question, top_k=3):
    try:
        question_embedding = get_embedding(question)
        cursor.execute("""
            SELECT content
            FROM document_chunks
            ORDER BY embedding <=> %s::vector  -- Add ::vector cast
            LIMIT %s
        """, (question_embedding, top_k))
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

if question:
    st.write(f"**Embedding for the question** '{question}':")
    question_embedding = get_embedding(question)
    st.write(question_embedding[:5] + ["..."])
    
    relevant_chunks = get_relevant_chunks(question)
    st.write("Top relevant chunks retrieved:")
    for i, chunk in enumerate(relevant_chunks, start=1):
        st.write(f"{i}. {chunk}")

    # Step 4: Generate Answer
    st.header("Step 4: Generate answer using Gemini")
    context = "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(relevant_chunks))
    prompt = f"""Using the following information:
{context}

Answer the question: {question}

Provide a clear and concise answer based only on the information provided above."""
    
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        st.write("**Generated Answer:**")
        st.write(answer)
    except Exception as e:
        st.error(f"Generation error: {str(e)}")

# Clean up
cursor.close()
conn.close()
