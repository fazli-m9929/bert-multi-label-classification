import streamlit as st
from ..inference import (
    load_model,
    predict_sorted,
    compute_hierarchy_probs,
    section_map, division_map, group_map
)
import pandas as pd

# Cache the tokenizer and model so they're loaded only once
@st.cache_resource
def load_resources():
    """
    Load and cache the tokenizer and model to avoid reloading on every rerun.

    Returns:
        tuple: (tokenizer, model)
    """
    return load_model()

def run():
    """
    Render the Streamlit UI, collect input text and threshold from the user,
    run prediction on button click, and display sorted label probabilities.
    """

    # Load cached model/tokenizer
    tokenizer, model = load_resources()

    # Page title and intro text
    st.title("Multi-Label Text Classifier (BERT)")
    st.markdown("Paste some text below and get predicted labels.")

    # Inject custom CSS for RTL text area and monospace font
    st.markdown(
        """
        <style>
        textarea {
            direction: rtl;
            font-family: monospace;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Text input box (shorter height, RTL styling applied above)
    user_input = st.text_area("Input Text", height=150)

    # Threshold slider and predict button in one row
    col1, col2 = st.columns([4, 1])
    with col1:
        # Slider to control prediction threshold
        threshold = st.slider(
            "Prediction Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            label_visibility="collapsed"
        )
    with col2:
        # Full-width Predict button
        run_pred = st.button("Predict", use_container_width=True)

    # When button is clicked
    if run_pred:
        if not user_input.strip():
            # Handle empty input
            st.warning("Input text is empty. Please provide some text.")
            return

        # Run model inference
        with st.spinner("Running inference..."):
            predictions = predict_sorted(user_input, tokenizer, model, threshold)

        # Display results
        if not predictions:
            st.info("No labels passed the threshold.")
        else:
            st.subheader("Predicted Labels")
            section_probs, division_probs, group_probs = compute_hierarchy_probs(predictions)
            
            st.write(
                pd.merge(section_map, section_probs, on="Id", how="right").set_index('Id')
            )
            st.write(
                pd.merge(division_map, division_probs, on="Id", how="right").set_index('Id')
            )
            st.write(
                pd.merge(group_map, group_probs, on="Id", how="right").set_index('Id')
            )
            
    else:
        # Message shown before clicking the button
        st.subheader("Press the button to Predict ...")