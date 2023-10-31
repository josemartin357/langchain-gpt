import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🐕 MartinGPT - Test application that uses LangChain')
prompt = st.text_input('Enter your prompt below')

# Using prompt template to reduce size of user input
title_template = PromptTemplate(
    input_variables = ['topic'], # using topic for additional validation and raise exception if mismatch
    template = 'write me a script about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title'], 
    template = 'write me a script based on this title {title}'
)

# LLMs
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True) # setting verbose to true so output is more readable
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True) 
sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True) # chain run title_chain first to generate video title for script_chain 

# Showing response on screen if there is a response
if prompt:
    response = sequential_chain.run(prompt) # passing sequential chain as response
    st.write(response)

