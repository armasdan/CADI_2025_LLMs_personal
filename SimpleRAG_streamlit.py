import os
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
import streamlit as st

# Configuración de DeepSeek
DEEPSEEK_API_URL = "https://api.deepseek.com/query"  # Reemplaza con el endpoint correcto
DEEPSEEK_PUBLIC_KEY = "tu_clave_publica"  # Reemplaza con tu clave pública de DeepSeek

# Función para interactuar con DeepSeek
def query_deepseek(question, context=""):
    """
    Envía una consulta a la API de DeepSeek.
    """
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            json={"query": question, "context": context},
            headers={"Authorization": f"Bearer {DEEPSEEK_PUBLIC_KEY}"},
        )
        response.raise_for_status()
        return response.json().get("response", "No se obtuvo una respuesta de DeepSeek.")
    except Exception as e:
        return f"Error al consultar DeepSeek: {str(e)}"

# Función para procesar el archivo PDF
def process_pdf(file):
    """
    Convierte el contenido del PDF en texto.
    """
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Interfaz de Streamlit
st.title("Chatbot de PDF con DeepSeek")
st.write("Sube un archivo PDF y haz preguntas sobre su contenido.")

uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    # Procesar el PDF
    with st.spinner("Procesando el archivo PDF..."):
        pdf_text = process_pdf(uploaded_file)

    # Dividir el texto en fragmentos
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    # Crear el índice FAISS
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    vectorstore = FAISS.from_texts(texts, embeddings)

    st.success("¡Archivo PDF procesado! Ahora puedes hacer preguntas.")

    # Entrada del usuario
    question = st.text_input("Haz tu pregunta:")
    if question:
        with st.spinner("Consultando a DeepSeek..."):
            # Recuperar contexto del índice FAISS
            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(question)
            context = " ".join([doc.page_content for doc in docs])

            # Realizar la consulta a DeepSeek
            response = query_deepseek(question, context)
            st.write(f"**Respuesta:** {response}")
