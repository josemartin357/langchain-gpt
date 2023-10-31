import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🐕 MartinGPT - Test application that uses LangChain')
prompt = st.text_input('Enter your prompt below')

# LLMs
llm = OpenAI(temperature=0.9)

# Showing response on screen if there is a response
if prompt:
    response = llm(prompt)
    st.write(response)

