import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🐕 MartinGPT - Test application that uses LangChain')
prompt = st.text_input('Enter your prompt below')

# Using prompt template to reduce size of user input
title_template = PromptTemplate(
    input_variables = ['topic'], # using topic for additional validation and raise exception if mismatch
    template = 'write me a script about {topic}'
)

# LLMs
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True) # setting verbose to true so output is more readable


# Showing response on screen if there is a response
if prompt:
    response = title_chain.run(topic=prompt)
    st.write(response)

