import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Función para leer el contenido del PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Procesar y cargar el PDF en FAISS
def process_pdf_with_faiss(pdf_text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(pdf_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

# Simulación de procesamiento IA
def get_relevant_response(question, vectorstore):
    """Obtiene una respuesta relevante utilizando FAISS."""
    if not vectorstore:
        return "No se ha cargado ningún documento para analizar. Por favor, sube un archivo PDF."

    docs = vectorstore.similarity_search(question, k=3)
    if docs:
        response = "\n\n".join([f"- {doc.page_content.strip()}" for doc in docs])
        return f"He encontrado información relevante en el documento:\n\n{response}"
    else:
        return "Lo siento, no he encontrado información relevante en el documento para responder tu pregunta."

# Interfaz de Streamlit
st.title("Chatbot del Reglamento del Laboratorio (IA Powered)")
st.write("Sube un PDF y haz preguntas relacionadas con su contenido. Este chatbot utiliza inteligencia artificial para responder con precisión.")

# Cargar archivo PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF para análisis adicional", type="pdf")

vectorstore = None
if uploaded_file:
    with st.spinner("Procesando el archivo PDF..."):
        pdf_text = read_pdf(uploaded_file)
        if pdf_text.strip():
            vectorstore = process_pdf_with_faiss(pdf_text)
            st.success("¡PDF procesado! Ahora puedes hacer preguntas basadas en el documento.")
        else:
            st.error("El PDF está vacío o no contiene texto extraíble.")

# Entrada del usuario
question = st.text_input("Haz una pregunta:")

# Responder preguntas utilizando el modelo
if question:
    with st.spinner("Procesando con inteligencia artificial..."):
        answer = get_relevant_response(question, vectorstore)
        st.write("**Respuesta:**")
        st.markdown(answer)
