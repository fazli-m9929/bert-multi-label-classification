from streamlit_app.pages.classifier import run as classifier_page
from streamlit_app.pages.classifier import description as classifier_description
from streamlit_app.pages.home import description as home_description
from streamlit_app.pages.home import run as home_page
import streamlit as st

def main():
    # Set global page configuration
    st.set_page_config(
        page_title="Company Labeler",
        page_icon="🧠",
        layout='wide'
    )

    # Define available pages and their corresponding run functions
    pages = {
        "🏠 Home": home_page,
        "🧪 Classifier Demo": classifier_page,
        # "🗺️ Mapping Demo": mapping_page,
        # "📊 DataFrame Explorer": dataframe_page,
    }
    
    descriptions = {
        "🏠 Home": home_description,
        "🧪 Classifier Demo": classifier_description,
        # "🗺️ Mapping Demo": mapping_page,
        # "📊 DataFrame Explorer": dataframe_page,
    }

    # Sidebar for selecting active page
    st.sidebar.title("🔍 Navigation")
    selected_page = st.sidebar.selectbox("Choose a section", list(pages.keys()))

        # A short description for sidebar
    st.sidebar.markdown("📝 **About this page:**")
    st.sidebar.markdown(f"**{selected_page} Page**")

    st.sidebar.markdown(
        f"""
        <div style="text-align: justify; font-size: 0.9rem;">
            {descriptions[selected_page]}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Render the selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()