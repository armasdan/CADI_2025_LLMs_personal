from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st
import os

# Cargar variables de entorno
load_dotenv()

# Configurar el modelo OpenAI
llm = OpenAI(temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY"))

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

# Configurar la memoria y la cadena de conversación
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

# Interfaz de Streamlit
st.header('My Chatbot')
st.write("Hola, estoy aquí para ayudarte.")
history = []

# Entrada del usuario
question = st.chat_input("Pregúntame algo")
if question:
    st.write(f"**Tú:** {question}")
    try:
        result = conversation_chain.invoke({"question": question, "chat_history": history})
        st.write(f"**Bot:** {result['answer']}")
        history.append((question, result["answer"]))
    except Exception as e:
        st.error(f"Error procesando tu pregunta: {str(e)}")
