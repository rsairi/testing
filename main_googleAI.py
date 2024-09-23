import os
import streamlit as st
import pickle
import time

from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("URLBot:URLs Info Extraction Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(1):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_googleai.pkl"

main_placeholder = st.empty()

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    instructor_embeddings = HuggingFaceEmbeddings()
    
    vectorstore_openai = FAISS.from_documents(docs, instructor_embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    #time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
print ("QUERY ####", query)

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            retriever = vectorstore.as_retriever()

            #chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            #chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, input_key="query", return_source_documents=False)

            chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)

            #result = chain({"question": query}, return_only_outputs=True)
            
            res = chain(query)
            
            print("result ####", res)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(res["result"])
            #st.write(result["answer"])

            # Display sources, if available
            sources = res.get("source_documents", "")

            if sources:
               for source in sources:
                    for i in source:
                        print ("INFO ####", i)     

            #sources = res.get("source", "")

            #if sources:
            #    st.subheader("Sources:")
            #    sources_list = sources.split("\n")  # Split the sources by newline
            #    for source in sources_list:
            #        st.write(source)
