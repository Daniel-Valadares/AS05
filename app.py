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

def update_values(list, documents):
    if documents:
        for document in documents:
            list.append(document.name)

def process_pdf(documents):
    important_info = ""
    for document in documents:
        with pdfplumber.open(document) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    important_info += f"\n--- Página {page_number} ---\n"
                    important_info += page_text.strip() + "\n"

                tables = page.extract_tables()
                if tables:
                    for table_index, table in enumerate(tables, start=1):
                        important_info += f"\nTabela {table_index} (Página {page_number}):\n"
                        for row in table:
                            row_text = " | ".join([cell.strip() if cell else "" for cell in row])
                            important_info += row_text + "\n"

    return important_info.strip()

def process_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def embed_text(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_texts(text_chunks, embeddings)
    vectors.save_local("faiss_index")

def prompt_template():
    base_template = """
    Por favor, responda à pergunta de forma detalhada e completa, utilizando as informações do contexto fornecido. Caso não seja possível responder com base no contexto, diga "Não consegui formular uma resposta :(".
    Contexto: 
    {context}

    Pergunta: 
    {question}

    Instruções: 
    - Seja direto, mas forneça explicações adicionais quando necessário.
    - Se a resposta envolver passos ou etapas, apresente-as de forma clara e ordenada.
    - Caso existam dúvidas ou ambiguidades na pergunta, apresente possíveis interpretações e responda de acordo.
    - Se o contexto apresentar números, tabelas ou dados, inclua-os na explicação.

    Resposta:

    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=base_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def answer_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = prompt_template()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    st.markdown("### Resposta Gerada:")
    st.success(response["output_text"])

def clear_cache():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

def main():
    st.set_page_config(page_title="Assistente LLM", layout="wide")

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar-header {
            font-size: 18px;
            font-weight: bold;
            color: #2E86C1;
        }
        .file-uploader {
            background-color: #E8F8F5;
            padding: 15px;
            border-radius: 10px;
        }
        .answer-box {
            background-color: #FBFCFC;
            padding: 15px;
            border: 1px solid #D5D8DC;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">Assistente Inteligente para PDFs</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-header">Configurações e Informações</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - **Modelo LLM:** Google Gemini-Pro  
            - **Embeddings:** LangChain Google API  
            - **Indexação de Texto:** FAISS  
            - **Frontend:** Streamlit  
            """
        )
        if st.button("Reiniciar Sessão"):
            clear_cache()

    documents = st.file_uploader("Envie seus documentos PDF", type=["pdf"], accept_multiple_files=True, help="Selecione arquivos PDF para processamento")
    file_names = ["Todos"]
    st.button("Carregar Arquivos", on_click=update_values(file_names, documents))

    selected_files = st.multiselect("Escolha os Arquivos:", file_names, placeholder="Selecione os documentos a usar")

    if documents:
        st.markdown('<div class="file-uploader">Arquivos enviados com sucesso!</div>', unsafe_allow_html=True)

    selected_documents = []
    if "Todos" in selected_files:
        selected_documents = documents
    else:
        selected_documents = [doc for doc in documents if doc.name in selected_files]

    if st.button("Processar Documentos"):
        with st.spinner("Processando..."):
            raw_text = process_pdf(selected_documents)
            chunks = process_chunks(raw_text, chunk_size=2000, chunk_overlap=500)
            embed_text(chunks)
        st.success("Documentos processados e indexados com sucesso!")

    user_question = st.text_input("Pergunte algo com base nos documentos:")
    if st.button("Obter Resposta"):
        with st.spinner("Gerando resposta..."):
            answer_question(user_question)

if __name__ == "__main__":
    main()