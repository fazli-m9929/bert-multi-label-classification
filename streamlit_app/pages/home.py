import streamlit as st

# Page sidebar description
description = (
    "Welcome to the Company Labeler App. "
    "<br><br>This tool showcases a multi-page platform for classifying company activity descriptions using a fine-tuned BERT model. "
    "<br>It maps free-text inputs to standardized industry labels based on a hierarchical taxonomy. "
    "<br><br>📌 Use the sidebar to navigate through available features and demos."
)

def run():
    """
    Render the home page of the Streamlit app.
    Shows an overview and guidance on navigating the multi-page demo.
    """

    # Set page title and subtitle
    st.title("🏠 Welcome to the Company Labeler App")
    st.markdown("""
        This application demonstrates a **multi-label text classification** system built using a fine-tuned BERT model.

        Use the sidebar on the left to:
        - Run the classifier demo
    
    """)
    with st.expander("ℹ️ How it works"):
        st.markdown(
        """
        - Input text is tokenized and passed into a trained BERT model.
        - The model returns probabilities for multiple labels.
        - You can set a prediction **threshold** to control which labels are returned.
        """
    )
    
    st.markdown("""
        - Feedback the model's performance on real data.
        
    """)
    with st.expander("ℹ️ How to Use the Feedback Page"):
        st.markdown(
            """
            1. From the sidebar on the left, select the **🧾 Feedback** page.
            2. Click the **🔄 Fetch a New Sample for Review** button to load a text sample with model predictions.
            3. Carefully read the displayed text.
            4. Review the predicted labels and check the boxes for all labels you think are correct.
            5. After selecting the correct labels, press the **✅ Submit Feedback** button.
            6. Your feedback will be saved, and the text will be marked as reviewed.
            7. You can fetch new samples and repeat the process to help improve model accuracy and performance.
            """
        )

    # Footer
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + PyTorch + Transformers")
