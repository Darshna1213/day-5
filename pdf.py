import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Streamlit app title
st.title("Chat with Your PDF 📄")

# Input for Google API Key
api_key = st.text_input("Enter your Google Gemini API Key", type="password")
if not api_key:
    st.warning("Please enter your Google Gemini API Key to proceed.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Process PDF and create vector store
if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("Processing PDF..."):
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        
        # Load and split PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
        
        # Initialize LLM and QA chain
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        prompt_template = """Use the provided context to answer the question concisely and accurately. If the answer is not in the context, say "I don't know."
        Context: {context}
        Question: {question}
        Answer: """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
        st.success("PDF processed! You can now ask questions.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the PDF"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response if QA chain is ready
    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please upload and process a PDF first.")