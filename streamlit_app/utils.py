from .inference import load_model
import streamlit as st

# Cache the tokenizer and model so they're loaded only once
@st.cache_resource
def load_resources():
    """
    Load and cache the tokenizer and model to avoid reloading on every rerun.

    Returns:
        tuple: (tokenizer, model)
    """
    return load_model()