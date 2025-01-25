import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Configuración de Groq
GROQ_API_KEY = "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"  # Reemplaza con tu clave de Groq
groq_model_name = "llama-3.1-70b-versatile"  # Modelo disponible en Groq

# Configurar Groq como LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=groq_model_name,
)

# Función para leer el contenido del PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Interfaz de Streamlit
st.title("Chatbot Inteligente para PDFs con Groq")
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
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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

            # Usar Groq para generar la respuesta
            response = llm.predict(context=f"Pregunta: {question}\nContexto: {context}")

        st.write("**Respuesta:**")
        st.write(response)
