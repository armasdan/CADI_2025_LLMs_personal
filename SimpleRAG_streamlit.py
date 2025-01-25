import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Configurar el modelo de embeddings de HuggingFace
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Configurar el modelo de generación de lenguaje HuggingFace
generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Función para leer el contenido del PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Interfaz de Streamlit
st.title("Chatbot Inteligente para PDFs (Gratis)")
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

        # Crear el índice FAISS utilizando los embeddings
        with st.spinner("Generando el índice de búsqueda..."):
            vectorstore = FAISS.from_texts(texts, embeddings)

    st.success("¡PDF procesado! Ahora puedes hacer preguntas.")

    # Entrada del usuario para hacer preguntas
    question = st.text_input("Haz una pregunta:")

    if question:
        with st.spinner("Buscando la respuesta..."):
            # Usar el índice FAISS para buscar los fragmentos más relevantes
            docs = vectorstore.similarity_search(question, k=3)
            context = " ".join([doc.page_content for doc in docs])

            # Usar el modelo de HuggingFace para generar la respuesta
            response = generator(
                f"Pregunta: {question}\nContexto: {context}\nRespuesta:",
                max_length=200,
                num_return_sequences=1,
            )[0]["generated_text"]

        st.write("**Respuesta:**")
        st.write(response)
