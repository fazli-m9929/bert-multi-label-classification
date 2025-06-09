import streamlit as st

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
        - View other components (coming soon)
        
        ---
    """)

    # Optional expandable help/instructions section
    with st.expander("ℹ️ How it works"):
        st.markdown(
        """
        - Input text is tokenized and passed into a trained BERT model.
        - The model returns probabilities for multiple labels.
        - You can set a prediction **threshold** to control which labels are returned.
        """
    )

    # Footer
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + PyTorch + Transformers")
