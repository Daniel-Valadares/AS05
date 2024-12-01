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
genai.configure(api_key=os.getenv("AIzaSyB8k3CqdZngSYKdcZH1RyBbOIdCYEvHTjg")) # Troque a string por uma chave

# Função para atualizar lista de arquivos
def append_file_names(file_list, uploaded_files):
    if uploaded_files:
        file_list.extend([file.name for file in uploaded_files])

# Função para processar PDFs
def extract_text_from_pdfs(uploaded_files):
    combined_text = ""
    for file in uploaded_files:
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_content = page.extract_text()
                if page_content:
                    combined_text += f"\n--- Página {i} ---\n{page_content.strip()}\n"
                for table_index, table in enumerate(page.extract_tables(), start=1):
                    combined_text += f"\nTabela {table_index} (Página {i}):\n"
                    combined_text += "\n".join([" | ".join(row or [""]) for row in table]) + "\n"
    return combined_text.strip()

# Função para dividir texto em blocos
def create_text_chunks(text, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# Função para gerar embeddings e salvar índice
def generate_embeddings(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index = FAISS.from_texts(chunks, embeddings)
    index.save_local("faiss_index")

# Template do prompt para o modelo
def build_prompt_chain():
    prompt_text = """
    Responda à pergunta com base no contexto fornecido. Caso o contexto não contenha a resposta, informe: "Não foi possível responder.".

    Contexto:
    {context}

    Pergunta:
    {question}

    Resposta:
    """
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
    return load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)

# Função para responder perguntas
def respond_to_question(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    documents = index.similarity_search(question)
    chain = build_prompt_chain()
    response = chain({"input_documents": documents, "question": question}, return_only_outputs=True)
    st.write("Resposta:", response["output_text"])

# Função para reiniciar o aplicativo
def reset_app():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

# Função principal
def main():
    st.set_page_config(page_title="Assistente LLM", layout="centered", initial_sidebar_state="collapsed")
    st.title("Assistente Inteligente para PDFs")
    st.sidebar.header("Configurações")

    if st.sidebar.button("Reiniciar Sessão"):
        reset_app()

    uploaded_files = st.file_uploader("Envie arquivos PDF", type=["pdf"], accept_multiple_files=True)
    file_names = ["Todos"]
    append_file_names(file_names, uploaded_files)

    selected_files = st.multiselect("Selecione os arquivos:", file_names, placeholder="Escolha os arquivos para o contexto")
    selected_documents = uploaded_files if "Todos" in selected_files else [file for file in uploaded_files if file.name in selected_files]

    if st.button("Processar"):
        with st.spinner("Processando documentos..."):
            raw_text = extract_text_from_pdfs(selected_documents)
            text_chunks = create_text_chunks(raw_text, chunk_size=2000, overlap=500)
            generate_embeddings(text_chunks)
        st.success("Documentos processados com sucesso!")

    question = st.text_input("Faça sua pergunta:")
    if st.button("Perguntar"):
        with st.spinner("Buscando resposta..."):
            respond_to_question(question)

if __name__ == "__main__":
    main()