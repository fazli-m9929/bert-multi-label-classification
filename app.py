from streamlit_app.pages.classifier import run as classifier_page
from streamlit_app.pages.home import run as home_page
import streamlit as st

if __name__ == "__main__":
    # Set global page configuration
    st.set_page_config(
        page_title="Company Labeler",
        page_icon="🧠",
    )

    # Define available pages and their corresponding run functions
    pages = {
        "🏠 Home": home_page,
        "🧪 Classifier Demo": classifier_page,
        # "🗺️ Mapping Demo": mapping_page,
        # "📊 DataFrame Explorer": dataframe_page,
    }

    # Sidebar for selecting active page
    st.sidebar.title("🔍 Navigation")
    selected_page = st.sidebar.selectbox("Choose a section", list(pages.keys()))

    # Render the selected page
    pages[selected_page]()