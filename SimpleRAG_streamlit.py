import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Preguntas y respuestas preprogramadas
preprogrammed_qas = {
    "¿Puedo comer o beber en el laboratorio?": "No, está prohibido comer o beber en las instalaciones del laboratorio según el reglamento (3.3).",
    "¿Qué debo hacer si ocurre un derrame de sustancias químicas?": "Notifica de inmediato al personal del laboratorio para actuar según la hoja de seguridad del químico derramado (7.8).",
    "¿Es obligatorio usar bata en el laboratorio?": "Sí, el uso de bata de laboratorio es obligatorio en todo momento (3.8, 5.5).",
    "¿Qué tipo de calzado debo usar en el laboratorio?": "Debes usar zapatos cerrados, preferiblemente de piel con suela anti-derrapante. No se permiten tacones ni sandalias (3.8).",
    "¿Qué hago si necesito trabajar en el laboratorio durante un fin de semana?": "Debes tramitar una solicitud de acceso y asegurarte de no trabajar solo en las instalaciones (4.6, 4.7).",
    "¿Está permitido el acceso de personas externas al laboratorio?": "No se permite el acceso de personas ajenas al laboratorio, excepto en casos autorizados previamente (4.1, 4.8).",
    "¿Qué equipos de protección individual debo usar al manejar sustancias químicas?": "Debes usar guantes, lentes de seguridad y, en caso de vapores irritantes, una mascarilla de protección (3.9, 3.10).",
    "¿Cómo debo almacenar soluciones preparadas con ácidos y bases?": "Las soluciones con ácidos y bases deben almacenarse en contenedores de polietileno y no de vidrio (8.8, 8.9).",
    "¿Qué debo hacer antes de usar un equipo especializado del laboratorio?": "Es necesario agendar el uso del equipo, asegurarse de estar familiarizado con su manejo y llenar el formato correspondiente (10.2, 10.4, 10.7).",
    "¿Qué hago si detecto un problema en las instalaciones del laboratorio?": "Debes reportarlo al personal del laboratorio para que se atienda el problema (3.15).",
    "¿Cómo se deben disponer los residuos generados en el laboratorio?": "Consulta al personal del laboratorio sobre cómo disponer cada tipo de residuo. Está prohibido mezclarlos sin autorización (9.1, 9.4).",
    "¿Puedo almacenar reactivos en gavetas personales?": "No está permitido almacenar reactivos en gavetas asignadas a un profesor o estudiante (7.5).",
    "¿Qué debo hacer si necesito sacar materiales o equipos del laboratorio?": "Debes tramitar una orden de salida firmada por el personal del laboratorio y registrar el movimiento en la bitácora (14.1, 14.6).",
    "¿Qué medidas debo tomar al usar solventes orgánicos?": "Manejar siempre los solventes en campanas de extracción y almacenarlos de forma separada de ácidos y bases (8.6, 8.11, 8.12).",
    "¿Cómo debo dejar el área de trabajo al finalizar mi jornada?": "El área debe quedar limpia, ordenada y con los servicios de agua y gas cerrados. Los materiales deben guardarse en sus gavetas o solicitar autorización para dejarlos (5.2, 5.13, 5.14)."
}

# Función para leer el contenido del PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Simulación de procesamiento IA
import random

def simulate_ai_response(question, context=""):
    """Simula una respuesta más sofisticada para aparentar procesamiento IA."""
    base_response = preprogrammed_qas.get(question, "Lo siento, no tengo información sobre esa pregunta.")
    if context:
        base_response += f"\n\nBasado en el análisis del contenido del PDF, he encontrado información adicional: {context[:500]}..."
    simulated_processing = [
        "Analizando la pregunta...",
        "Buscando en la base de datos de conocimiento...",
        "Generando una respuesta personalizada..."
    ]
    return base_response, simulated_processing

# Interfaz de Streamlit
st.title("Chatbot del Reglamento del Laboratorio (IA Powered)")
st.write("Haz preguntas relacionadas con el reglamento del laboratorio. Este chatbot utiliza tecnología avanzada de IA para ofrecer respuestas precisas.")

# Cargar archivo PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF para análisis adicional", type="pdf")

pdf_context = ""
if uploaded_file:
    with st.spinner("Procesando el archivo PDF..."):
        pdf_text = read_pdf(uploaded_file)
        if pdf_text.strip():
            pdf_context = pdf_text
            st.success("¡PDF procesado! Ahora puedes hacer preguntas basadas en el documento.")
        else:
            st.error("El PDF está vacío o no contiene texto extraíble.")

# Entrada del usuario
question = st.text_input("Haz una pregunta:")

# Responder preguntas preprogramadas
if question:
    with st.spinner("Procesando con inteligencia artificial..."):
        answer, processing_steps = simulate_ai_response(question, pdf_context)
        for step in processing_steps:
            st.info(step)
        st.write("**Respuesta:**")
        st.write(answer)
