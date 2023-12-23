import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("QA from CSV")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()
    
st.markdown(""" Example questions: 
            ( should I learn power bi or tableau? , I've a MAC computer. Can I use powerbi on it? ,do you have javascript course? )""")

question = st.text_input("Question: ", placeholder="Write Like: What is your refund policy?")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])
