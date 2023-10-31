import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# * App framework
st.title('🐕 MartinGPT - Test application that uses LangChain')
prompt = st.text_input('Enter your prompt below')

# * Using prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], # using topic for additional validation and raise exception if mismatch
    template = 'write me a script about {topic}'
)
script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template = 'write me a script based on this title {title} while leveraging this wikipedia reserch:{wikipedia_research}'
)

# * Memory use for history
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# *LLMs
llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory) # setting verbose to true so output is more readable

script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

# sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'] ,verbose=True) # chain run title_chain first to generate video title for script_chain. SequentialChain outputs multiple chain of outputs

# sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True) # chain run title_chain first to generate video title for script_chain. SimpleSequentialChain will output the last output

wiki = WikipediaAPIWrapper() # new instance of wikipedia api wrapper


# * Showing response on screen if there is a response
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
