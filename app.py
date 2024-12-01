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


load_dotenv()

# Configurar chave de API do Google
google_api_key = os.getenv("AIzaSyB8k3CqdZngSYKdcZH1RyBbOIdCYEvHTjg")
if not google_api_key:
    st.error("Chave de API do Google não configurada. Verifique o arquivo .env.")
else:
    genai.configure(api_key=google_api_key)


def update_values(file_list, documents):
    """Atualiza a lista de arquivos carregados."""
    if documents:
        for document in documents:
            file_list.append(document.name)


def process_pdf(documents):
    """Processa documentos PDF e extrai texto e tabelas."""
    if not documents:
        return ""

    important_info = ""
    try:
        for document in documents:
            with pdfplumber.open(document) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    # Extrair texto limpo da página
                    page_text = page.extract_text()
                    if page_text:
                        important_info += f"\n--- Página {page_number} ---\n"
                        important_info += page_text.strip() + "\n"

                    # Extrair tabelas e formatar como texto
                    tables = page.extract_tables()
                    if tables:
                        for table_index, table in enumerate(tables, start=1):
                            important_info += f"\nTabela {table_index} (Página {page_number}):\n"
                            for row in table:
                                row_text = " | ".join([cell.strip() if cell else "" for cell in row])
                                important_info += row_text + "\n"

        return important_info.strip()
    except Exception as e:
        st.error(f"Erro ao processar PDFs: {e}")
        return ""


def process_chunks(text, chunk_size, chunk_overlap):
    """Divide o texto em fragmentos menores."""
    if not text:
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Erro ao dividir texto em fragmentos: {e}")
        return []


def embed_text(text_chunks):
    """Gera embeddings a partir dos fragmentos de texto."""
    if not text_chunks:
        st.error("Nenhum texto para gerar embeddings.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectors = FAISS.from_texts(text_chunks, embeddings)
        vectors.save_local("faiss_index")
    except Exception as e:
        st.error(f"Erro ao gerar embeddings: {e}")


def prompt_template():
    """Cria o modelo de prompt para geração de respostas."""
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
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
        prompt = PromptTemplate(template=base_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Erro ao configurar o modelo: {e}")
        return None


def answer_question(user_question):
    """Gera uma resposta com base na pergunta do usuário."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)
        chain = prompt_template()

        if not chain:
            st.error("Erro na inicialização do modelo.")
            return

        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Resposta:", response["output_text"])
    except Exception as e:
        st.error(f"Erro ao gerar resposta: {e}")


def main():
    """Interface principal da aplicação."""
    st.set_page_config(page_title="Assistente Inteligente", layout="centered")

    st.title('Interprete PDFs com Inteligência Artificial')
    st.subheader("Faça perguntas baseadas nos documentos carregados!")

    documents = st.file_uploader("Carregue seus documentos PDF aqui:", type=['pdf'], accept_multiple_files=True)

    file_names = ['Todos']
    st.button("Enviar", on_click=update_values, args=(file_names, documents))

    selected_files = st.multiselect("Escolha os arquivos para análise:", file_names, placeholder="Selecione os documentos")

    selected_documents = []
    if 'Todos' in selected_files:
        selected_documents = documents
    else:
        for document in documents:
            if document.name in selected_files:
                selected_documents.append(document)

    chunk_size = len(selected_documents) * 2000
    chunk_overlap = int(chunk_size * 0.25)

    user_input = st.text_input("Digite sua pergunta aqui:")
    if st.button("Responder"):
        with st.spinner("Processando..."):
            raw_text = process_pdf(selected_documents)
            text_chunks = process_chunks(raw_text, chunk_size, chunk_overlap)
            embed_text(text_chunks)
        answer_question(user_input)


if __name__ == '__main__':
    main()
