import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

# Configurar HuggingFace como LLM
huggingface_api_key = None  # Cambia a tu clave API si tienes una, o déjalo como `None` para usar modelos públicos.
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # Modelo HuggingFace
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=huggingface_api_key,
)

# Función para leer el contenido del PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Interfaz de Streamlit
st.title("Chatbot Inteligente para PDFs con HuggingFace")
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
            try:
                # Usar el índice FAISS para buscar los fragmentos más relevantes
                docs = vectorstore.similarity_search(question, k=3)
                context = " ".join([doc.page_content for doc in docs])

                # Crear el input como texto
                input_text = f"Contexto: {context}\nPregunta: {question}"

                # Usar HuggingFace para generar la respuesta
                response = llm(input_text)
                st.write("**Respuesta:**")
                st.write(response)
            except Exception as e:
                st.error(f"Error procesando la pregunta: {str(e)}")
