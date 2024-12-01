import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_js_eval import streamlit_js_eval

# Configuração inicial
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def append_filenames(file_list, uploaded_files):
    if uploaded_files:
        for file in uploaded_files:
            file_list.append(file.name)

def extract_text_from_pdfs(uploaded_files):
    extracted_text = ""
    for file in uploaded_files:
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    extracted_text += f"\n--- Página {page_num} ---\n{page_text.strip()}\n"
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables, start=1):
                        extracted_text += f"\nTabela {table_idx} (Página {page_num}):\n"
                        for row in table:
                            extracted_text += " | ".join([cell.strip() if cell else "" for cell in row]) + "\n"
    return extracted_text.strip()

def split_text_into_chunks(content, size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(content)

def generate_embeddings(chunks):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store.save_local("faiss_index")

def build_qa_chain():
    template = """
    Responda à pergunta com base no contexto abaixo. Caso não seja possível responder, diga "A resposta não pôde ser formulada. Por favor, forneça mais contexto ou materiais adicionais para consulta!".
    Contexto:
    {context}
    Pergunta:
    {question}
    Resposta:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    return load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)

def provide_answer(question):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    related_docs = vector_store.similarity_search(question)
    qa_chain = build_qa_chain()
    response = qa_chain({"input_documents": related_docs, "question": question}, return_only_outputs=True)
    st.write("Resposta:", response["output_text"])

def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

def app():
    st.set_page_config(page_title="Assistente PDF", layout="centered")
    st.title("Analise PDFs com IA")
    st.subheader("Envie seus arquivos e faça perguntas baseadas no conteúdo!")

    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 500

    uploaded_files = st.file_uploader("Carregue PDFs:", type=["pdf"], accept_multiple_files=True)
    file_list = ["Todos"]
    st.button("Atualizar Lista", on_click=append_filenames, args=(file_list, uploaded_files))
    
    chosen_files = uploaded_files

    chunk_size = len(chosen_files) * 2000
    chunk_overlap = chunk_size * 0.25

    question = st.text_input("Escreva sua pergunta:")
    if st.button("Gerar Resposta"):
        with st.spinner("Processando..."):
            full_text = extract_text_from_pdfs(chosen_files)
            text_chunks = split_text_into_chunks(full_text, chunk_size, chunk_overlap)
            generate_embeddings(text_chunks)
        provide_answer(question)
        st.success("Finalizado!")

if __name__ == "__main__":
    app()