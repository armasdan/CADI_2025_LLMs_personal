import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# Configuración del modelo OpenAI
OPENAI_API_KEY = os.getenv("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")  # Reemplaza con tu clave válida
llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)

# Función para procesar el archivo PDF
def process_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Interfaz de Streamlit
st.title("Chatbot de PDF")
st.write("Sube un archivo PDF y haz preguntas sobre su contenido.")

uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    # Procesar el PDF
    with st.spinner("Leyendo el archivo PDF..."):
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
