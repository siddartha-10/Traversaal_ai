from langchain_experimental.agents import create_csv_agent
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os
load_dotenv()
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import JSONLoader
import requests
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


llm = AzureChatOpenAI(openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2023-07-01-preview"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt4chat"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://gpt-4-trails.openai.azure.com/"),
        api_key=os.environ.get("AZURE_OPENAI_KEY"))


def metadata_func(record: str, metadata: dict) -> dict:
    lines = record.split('\n')
    locality_line = lines[10]
    price_range_line = lines[12]
    locality = locality_line.split(': ')[1]
    price_range = price_range_line.split(': ')[1]
    metadata["location"] = locality
    metadata["price_range"] = price_range

    return metadata

# Instantiate the JSONLoader with the metadata_func
jq_schema = '.parser[] | to_entries | map("\(.key): \(.value)") | join("\n")'
loader = JSONLoader(
    jq_schema=jq_schema,
    file_path='data.json',
    metadata_func=metadata_func,
)

# Load the JSON file and extract metadata
documents = loader.load()


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # Check if the FAISS index file already exists
    if os.path.exists("faiss_index"):
        # Load the existing FAISS index
        vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings)
        print("Loaded existing FAISS index.")
    else:
        # Create a new FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
        # Save the new FAISS index locally
        vectorstore.save_local("faiss_index")
        print("Created and saved new FAISS index.")
    return vectorstore

#docs = new_db.similarity_search(query)

vector = get_vectorstore(documents)


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory

template = """

context:- I have low budget what is the best hotel in Instanbul?
anser:- The other hotels in instanbul are costly and are not in your budget. so the best hotel in instanbul for you is hotel is xyz."

Don’t give information not mentioned in the CONTEXT INFORMATION. 
The system should take into account various factors such as location, amenities, user reviews, and other relevant criteria to 
generate informative and personalized explanations.
{context} 
Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context","question"])

chain_type_kwargs = {"prompt": prompt}
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
)


def main():
    st.title("Hotel Assistant Chatbot")
    st.write("Welcome to the Hotel Assistant Chatbot!")
    user_input = st.text_input("User Input:")
    
    if st.button("Submit"):
        response = chain.run(user_input)
        st.text_area("Chatbot Response:", value=response)
    
    if st.button("Exit"):
        st.stop()

if __name__ == "__main__":
    main()