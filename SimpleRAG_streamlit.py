import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Configuración del modelo HuggingFace
HUGGINGFACE_API_TOKEN = "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"  # Reemplaza con tu token de HuggingFace
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xl",  # Modelo de HuggingFace
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
)

# Función para procesar el PDF
def process_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Interfaz de Streamlit
st.title("Chatbot de PDF (HuggingFace)")
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

    # Configurar la memoria y la cadena conversacional
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )

    st.success("¡Archivo PDF procesado! Ahora puedes hacer preguntas.")

    # Entrada del usuario
    question = st.text_input("Haz tu pregunta:")
    if question:
        with st.spinner("Pensando..."):
            response = conversation_chain({"question": question})
            st.write(f"**Respuesta:** {response['answer']}")
