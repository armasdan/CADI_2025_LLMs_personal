import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Modelo de embeddings de HuggingFace (local y gratuito)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Función para leer el contenido del PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Interfaz de Streamlit
st.title("Chatbot Inteligente para PDFs")
st.write("Sube un archivo PDF y haz preguntas sobre su contenido.")

# Cargar archivo PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    # Leer y procesar el PDF
    with st.spinner("Procesando el archivo PDF..."):
        pdf_text = read_pdf(uploaded_file)

        # Dividir el contenido del PDF en fragmentos manejables
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(pdf_text)

        # Cargar el modelo de embeddings
        with st.spinner("Generando el índice de búsqueda..."):
            model = SentenceTransformer(EMBEDDING_MODEL)
            embeddings = model.encode(texts)

            # Crear índice FAISS
            vectorstore = FAISS.from_texts(texts, model)

    st.success("¡PDF procesado! Ahora puedes hacer preguntas.")

    # Entrada del usuario para hacer preguntas
    question = st.text_input("Haz una pregunta:")

    if question:
        with st.spinner("Buscando la respuesta..."):
            # Usar el índice FAISS para buscar el fragmento más relevante
            docs = vectorstore.similarity_search(question, k=3)
            response = " ".join([doc.page_content for doc in docs])

        st.write("**Respuesta:**")
        st.write(response)
