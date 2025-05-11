import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from dotenv import load_dotenv


load_dotenv()

st.title("News Research Tool")
st.sidebar.title("URL's to search")
file_path = "vector_database.pkl"
place_holder = st.empty()
urls = []
llm = Ollama(model='llama2')


for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url) 

search_buttom = st.sidebar.button("Process URL's")    

if search_buttom:
    loader = UnstructuredURLLoader(urls=urls)
    place_holder.text("Data Loading started......")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
       separators=['\n\n', '\n', ',', '.'],
       chunk_size = 500,
       chunk_overlap = 100)
    place_holder.text("Text splitter started......")

    docs = text_splitter.split_documents(data)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")     
    vector_store = FAISS.from_documents(docs, embedding_model)
    place_holder.text("Embedding text started......")

    with open(file_path, 'wb') as f:
      pickle.dump(vector_store, f)

query = st.text_input("Question : ")

if query:
   if os.path.exists(file_path):
      with open(file_path, "rb") as f:
         vector_store = pickle.load(f)
         chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vector_store.as_retriever())
         place_holder.text("Gathering information.......")
         result = chain({"question": query}, return_only_outputs=True)
         st.header("Answer")
         st.subheader(result["answer"])

         # display sources if available
         sources = result.get("Sources", "")
         if sources:
            st.subheader("Sources")
            sources.list = sources.split('\n')
            for source in sources:
               st.write(source)


# https://www.cnbctv18.com/market/stocks/tata-motors-share-price/TM03/               
               