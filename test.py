import os
from dotenv import load_dotenv
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# import google.generativeai

# Load environment variables from .env
load_dotenv()
# Retrieve the API key from the environment

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = GooglePalm(google_api_key=google_api_key, temperature=0.6)

# poem = llm("Write a 4 line poem of my love for samosa")
# print(poem)

loder = CSVLoader(file_path="file/collection.csv", source_column="prompt")

docs = loder.load()

# print(docs)

embeddings = HuggingFaceEmbeddings()

# e = embeddings.embed_query("What is your refund policy?")

# print(e[:5])

vector_db = FAISS.from_documents(documents=docs, embedding=embeddings)

retriver = vector_db.as_retriever()

# rdocs = retriver.get_relevant_documents("What is your refund policy?")

# print(rdocs)

prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}


chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriver,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)


resp = chain("What is your refund policy?")

print(resp)
