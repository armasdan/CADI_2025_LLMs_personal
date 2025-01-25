import streamlit as st
from PyPDF2 import PdfReader

# Función para leer el contenido del PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Interfaz de Streamlit
st.title("Chatbot PDF")
st.write("Sube un archivo PDF y haz preguntas sobre su contenido.")

# Cargar archivo PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    # Leer y mostrar el contenido del PDF
    with st.spinner("Leyendo el archivo PDF..."):
        pdf_text = read_pdf(uploaded_file)

    st.success("PDF procesado correctamente. Puedes hacer preguntas.")

    # Entrada del usuario para hacer preguntas
    question = st.text_input("Haz una pregunta:")

    if question:
        with st.spinner("Pensando..."):
            # Responder la pregunta de forma simple buscando palabras clave en el texto
            if question.lower() in pdf_text.lower():
                st.write("**Respuesta:** Parece que tu pregunta está en el PDF.")
            else:
                st.write("**Respuesta:** No encontré información relacionada en el PDF.")
