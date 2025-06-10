import streamlit as st
from ..inference import (
    predict_sorted,
    compute_hierarchy_probs,
    THRESHOLD_DEFAULT,
    section_map, division_map, group_map
)
from ..utils import load_resources
import pandas as pd

# Page sidebar description
description = (
    "This page allows you to classify a company's domain of activity based on free-text descriptions. "
    "A fine-tuned BERT model performs multi-label classification and outputs the most likely industry groups, "
    "as well as their parent divisions and sections using a hierarchical probability scheme."
    "<br><br>💡 Paste a description and click <strong>Predict</strong> to see the top categories."
)

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
            text-align: justify;
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
            min_value=0.2,
            max_value=0.8,
            value=THRESHOLD_DEFAULT,
            step=0.05,
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
            
            with st.spinner("Generating Results..."):
                section_probs, division_probs, group_probs, _ = compute_hierarchy_probs(predictions)
            
            st.write("Sections:")
            st.dataframe(
                pd.merge(section_map, section_probs, on="Id", how="right").set_index('Id')
            )
            
            st.write("Divisions:")
            st.dataframe(
                pd.merge(division_map, division_probs, on="Id", how="right").set_index('Id')
            )
            
            st.write("Groups:")
            st.dataframe(
                pd.merge(group_map, group_probs, on="Id", how="right").set_index('Id')
            )
            
    else:
        # Message shown before clicking the button
        st.subheader("Press the button to Predict ...")