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
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory

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


def api_call(text):
  url = "https://api-ares.traversaal.ai/live/predict"

  payload = { "query": [text]}
  headers = {
    "x-api-key": "ares_a0866ad7d71d2e895c5e05dce656704a9e29ad37860912ad6a45a4e3e6c399b5",
    "content-type": "application/json"
  }

  response = requests.post(url, json=payload, headers=headers)

  # here we will use the llm to summarize the response received from the ares api
  response_data = response.json()
  #print(response_data)
  try:
    response_text = response_data['data']['response_text']
    web_urls = response_data['data']['web_url']
    # Continue processing the data...
  except KeyError:
    print("Error: Unexpected response from the API. Please try again or contact the api owner.")
    # Optionally, you can log the error or perform other error handling actions.
  

  if len(response_text) > 10000:
    response_text = response_text[:8000]
    prompt = f"Summarize the following text in 500-100 0 words and jsut summarize what you see and do not add anythhing else: {response_text}"
    summary = llm.invoke(prompt)
    print(summary)
  else:
    summary = response_text

  result = "{} My list is: {}".format(response_text, web_urls)

# Convert the result to a string
  result_str = str(result)

  return result_str


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



prompt = """Please write the response to the user query: using the final_response and api_resource and make sure you are
The system should take into account various factors such as location, amenities, user reviews, and other relevant criteria to 
generate informative and personalized explanations. Do not add any information that is not mentioned in the context.
and make sure the answer is up to the point and not too long.

question: when did sachin hit his 100th century?
final_response: I can you assist you with hotel's or travels or food but cannot help other than that..

"""


def main():
    st.title("Travel Assistant Chatbot JR")
    st.write("Welcome to the Travel Assistant Chatbot!")
    user_input = st.text_input("User Input:")
    
    if st.button("Submit"):
        response = chain.run(user_input)
        api_response = api_call(user_input)
        response = llm.invoke(prompt+user_input+response + api_response)
        st.text_area("Chatbot Response:", value=response.content)
    
    if st.button("Exit"):
        st.stop()

if __name__ == "__main__":
    main()