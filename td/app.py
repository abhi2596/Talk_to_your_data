from langchain_unify import ChatUnify
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd 
import streamlit as st
import unify 

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

@st.cache_resource(experimental_allow_widgets=True)
def load_llm(api_key,model_name,provider_name):
    llm = ChatUnify(unify_api_key=api_key,model=f"{model_name}@{provider_name}")
    return llm

@st.experimental_fragment()
def clear_fragment():
    st.button("Clear Chat History",on_click=reset)

with st.sidebar:
    api_key,model_name,provider_name = mp_fragment()
    llm = load_llm(api_key,model_name,provider_name)

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", accept_multiple_files=False,on_change=reset)

if uploaded_file is not None:
    @st.cache_resource
    def agent_load(csv_file,_llm):
        df = pd.read_csv(csv_file)
        print(df.shape)
        agent = create_pandas_dataframe_agent(
                    _llm,
                    df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                )
        return agent
    agent = agent_load(uploaded_file,llm)

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
            response = agent.invoke(prompt)
            response = st.write(response)
            # response = st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})