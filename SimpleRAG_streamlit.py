import requests
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Configurar la URL de la API pública de DeepSeek
DEEPSEEK_API_URL = "https://api.deepseek.com/query"  # Asegúrate de tener el endpoint correcto
DEEPSEEK_PUBLIC_KEY = "tu_clave_publica"  # Reemplaza con tu clave pública de DeepSeek

# Función para interactuar con DeepSeek
def query_deepseek(question, context=""):
    """
    Realiza una consulta a la API pública de DeepSeek.
    """
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            json={"query": question, "context": context},
            headers={"Authorization": f"Bearer {DEEPSEEK_PUBLIC_KEY}"},
        )
        response.raise_for_status()
        return response.json().get("response", "No se pudo obtener respuesta de DeepSeek.")
    except Exception as e:
        return f"Error al consultar DeepSeek: {str(e)}"

# Función para cargar FAISS o crear uno nuevo
def load_db(embeddings, pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile"
    )
    docs = text_splitter.split_text(text)
    vectorstore = FAISS.from_texts(docs, embeddings)
    return vectorstore

# Inicializar embeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')

# Ruta del archivo PDF
pdf_path = "PeterPan.pdf"
index_path = "faiss_index"

# Cargar o construir el índice FAISS
if not os.path.exists(index_path):
    vectorstore = load_db(embeddings, pdf_path)
    vectorstore.save_local(index_path)
else:
    vectorstore = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever()

# Plantilla del prompt
template = """Eres un asistente para contestar preguntas del usuario sobre el contenido del archivo subido, que trata de la historia de Peter Pan. 
Contesta siempre en español y agradece al usuario por preguntar. Si las preguntas son sobre otro tema, contesta que no puedes contestar.
{context}
Question: {question}
Helpful Answer:"""
qa_prompt = ChatPromptTemplate.from_template(template)

# Configurar la memoria para almacenar el historial
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Interfaz de Streamlit
st.header('My Chatbot')
st.write("Hola, estoy aquí para ayudarte.")
history = []

# Entrada del usuario
question = st.chat_input("Pregúntame algo")
if question:
    st.write(f"**Tú:** {question}")
    try:
        # Recuperar contexto del FAISS
        context_docs = retriever.get_relevant_documents(question)
        context = " ".join([doc.page_content for doc in context_docs])

        # Realizar la consulta a DeepSeek
        response = query_deepseek(question, context)
        st.write(f"**Bot:** {response}")
        history.append((question, response))
    except Exception as e:
        st.error(f"Error procesando tu pregunta: {str(e)}")
