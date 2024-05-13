from langchain_unify import ChatUnify
import streamlit as st
import unify 
import re 
from langchain_experimental.utilities import PythonREPL 

def reset():
    st.session_state.messages = []

st.title("Talk to your Data")


@st.cache_data(experimental_allow_widgets=True)
def get_api_key():
    api_key = st.text_input("Unify AI Key",type="password")
    return api_key

@st.cache_data(experimental_allow_widgets=True)
def provider(model_name):
    provider_name = st.selectbox("Select a Provider",options=unify.list_providers(model_name))
    return provider_name

@st.experimental_fragment
def mp_fragment():
    api_key = get_api_key()
    model_name = st.selectbox("Select Model",options=unify.list_models(),index=1)
    provider_name = provider(model_name)
    return api_key,model_name,provider_name


def load_llm(api_key,model_name,provider_name):
    llm = ChatUnify(unify_api_key=api_key,model=f"{model_name}@{provider_name}")
    return llm

def clear_fragment():
    st.button("Clear Chat History",on_click=reset)

with st.sidebar:
    api_key,model_name,provider_name = mp_fragment()
    llm = load_llm(api_key,model_name,provider_name)

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", accept_multiple_files=False,on_change=reset)

if uploaded_file is not None:
    with open("data/dataframe.csv","wb") as f:
        f.write(uploaded_file.getbuffer())

with st.sidebar:
    clear_fragment()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        initial = " Follow this steps before asking questions about your data \n 1. Enter Unify API key \n 2. Select a Model and Provider using the sidebar \n 3. Upload a csv file \n 4. Any changes to sidebar will reset chat"
        st.markdown(initial)
        st.session_state.messages.append({"role": "assistant", "content": initial})

# Accept user input
if prompt := st.chat_input():
    # Add user message to chat history
    if api_key is None:
        if uploaded_file is None:
            st.warning("Enter API Key and a Upload CSV file to start")
        else:
            st.warning("Enter a Unify API key to start")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            react_prompt = f"""
            You are a expert in Python and Pandas library for data analysis. 
            
            You have access to the following tool:
            Python REPL : useful for executing python code.
            Response To Human: When you need to respond to the human you are talking to.

            
            You will receive a message from the human, then you should start a loop and do one of two things
            
            Option 1: You use a tool to answer the question.
            For this, you should use the following format:
            Thought: you should always think about what to do
            Action: the action to take, should be [Python REPL]
            Action Input: the input to the action, to be sent to the tool this should be enclosed in python's print function always
            
            After this, the human will respond with an observation, and you will continue.
            
            Option 2: You respond to the human.
            For this, you should use the following format:
            Action: Response To Human
            Action Input: your response to the human, summarizing what you did and what you learned
            
            Example:
            Human Input: Give details about first five rows of dataframe
            Thought: for this we need to execute this code print(df.head())
            Action: Python REPL tool needs to be used
            Action Input: print(df.head())

            Begin!
            Human Input: {prompt}
            """
            python_repl = PythonREPL()
            output = llm.invoke(react_prompt).content
            pattern = r"Action Input: (.+)"
            match = re.search(pattern, output)
            if match:
                action_input = match.group(1)
                print("Action Input:", action_input)
            else:
                print("No match found.")
            text = f"""import pandas as pd\ndf=pd.read_csv('data/dataframe.csv')\n{action_input}"""
            response_df = python_repl.run(text)
            response = st.write(response_df)
            # response = st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
