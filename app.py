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
os.getenv("GOOGLE_API_KEY")
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
    Por favor, responda à pergunta de forma detalhada e completa, utilizando as informações do contexto fornecido. Caso não seja possível responder com base no contexto, diga "Não consegui formular uma resposta...".
    Contexto: 
    {context}

    Pergunta: 
    {question}

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
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True)

    st.write("Resposta:", response["output_text"])


def clear_cache():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


def main():
    PAGE_CONFIG = {"page_title": "Assistente Inteligente", "layout": "centered"}
    st.set_page_config(**PAGE_CONFIG)

    st.title('Interprete PDFs com Inteligência Artificial')
    st.subheader("Faça perguntas baseadas nos documentos carregados!")

    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 500

    documents = st.file_uploader("Carregue seus documentos PDF aqui:", type=['pdf'], accept_multiple_files=True)

    file_names = ['Todos']

    st.button("Enviar", on_click=update_values(file_names, documents))

    selected_files = st.multiselect("Escolha os arquivos para análise:", file_names, placeholder="Selecione os documentos")

    selected_documents = []

    if 'Todos' in selected_files:
        selected_documents = documents
    else:
        for document in documents:
            if document.name in selected_files:
                selected_documents.append(document)

    chunk_size = len(selected_documents) * 2000
    chunk_overlap = chunk_size * 0.25

    user_input = st.text_input("Digite sua pergunta aqui:")
    if st.button("Responder"):
        with st.spinner("Processando..."):
            raw_text = process_pdf(selected_documents)
            text_chunks = process_chunks(raw_text, chunk_size, chunk_overlap)
            embed_text(text_chunks)
        answer_question(user_input)
        st.success("Processamento concluído!")


if __name__ == '__main__':
    main()
