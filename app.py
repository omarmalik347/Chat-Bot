import asyncio
import streamlit as st
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# App title
st.set_page_config(page_title="GenAI Chat Bot")

# Hugging Face Credentials
with st.sidebar:
    st.title('GenAI Chat Bot')
    
    # Use Hugging Face API Key from GitHub Secrets
    api_key = os.getenv("LLAMA3")

# Ensure there's an asyncio loop
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

if not api_key:
    st.error("API key is missing!")
    st.stop()

# Authenticate with Hugging Face Hub
try:
    login(api_key)
    st.success('API successfully authenticated!', icon='✅')
except Exception as e:
    st.error(f"Authentication failed: {e}")
    st.stop()

st.subheader('Models and Parameters')

# Model selection categories
model_options = {
    "Basic": ["meta-llama/Llama-3.2-1B"],
    "Basic-Medium": ["meta-llama/Llama-3.2-1B-Instruct"],
    "Medium-Fine": ["meta-llama/Llama-3.2-3B"],
    "Finest": ["meta-llama/Llama-3.2-3B-Instruct"]
}

# Select category (default set to "Basic-Medium")
selected_category = st.sidebar.selectbox('Select Model Category', list(model_options.keys()), index=1)
selected_model = model_options[selected_category][0]

# Slider inputs for parameters
temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.3, step=0.01)
top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
max_length = st.sidebar.slider('max_length', min_value=20, max_value=80, value=65, step=5)

st.markdown("Disclaimer: The performance and speed of this GenAI tool depend on the machine configuration and model selection")

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Load the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    model = AutoModelForCausalLM.from_pretrained(selected_model, torch_dtype=torch.bfloat16, device_map="auto")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function for generating response using Hugging Face model
def generate_huggingface_response(prompt_input):
    inputs = tokenizer(prompt_input, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_new_tokens=max_length, temperature=temperature, top_p=top_p, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Oops! Something went wrong."

# User-provided prompt
if prompt := st.chat_input(disabled=not api_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_huggingface_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
