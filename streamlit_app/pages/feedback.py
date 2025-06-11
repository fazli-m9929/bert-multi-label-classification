import streamlit as st
from ..utils import load_resources
from ..inference import (
    predict_sorted,
    compute_hierarchy_probs,
    THRESHOLD_DEFAULT,
    section_dict, division_dict,
    group_dict, label_dict_inv
)
import time
import sqlalchemy

# Page sidebar description
description = (
    "Help us improve the classifier by providing real-world feedback. "
    "<br><br>On this page, you'll review a sample prediction and tell us whether the model got it right. "
    "<br>You can also leave comments to highlight errors or edge cases."
    "<br><br>📌 Your input helps us evaluate and improve system performance over time."
)

def run():
    """
    Render the Feedback page of the Streamlit app.
    Get the User feedback on labels and store them in database for evaluations
    """
    
    st.title("🧾 Feedback")
    st.markdown(
        "Help us improve the classifier by evaluating predictions on real data."
    )


    # This runs only once per session and sets up our "memory".
    if 'sample_fetched' not in st.session_state:
        st.session_state.sample_fetched = False
        st.session_state.text_id = None
        st.session_state.text = ""
        st.session_state.predictions = []
        st.session_state.existing_feedback = set()
        
    # Uses the credentials from .streamlit/secrets.toml to connect to db
    conn = st.connection("postgresql", type="sql")
        
    # Load cached model/tokenizer
    tokenizer, model = load_resources()
    
    if st.button("🔄 Fetch a New Sample for Review", use_container_width=True):
        # The button's only job is to get data and save it to our "memory".
        
        # The fetch query
        query = """
            SELECT id, companyactivity FROM dbo.activity
            WHERE 	labels3 is not null AND 
                    labels3 not like '%[]%' AND 
                    is_reviewed is False
            ORDER BY RANDOM()
            LIMIT 1;
        """
        
        # Fallback if the query fails
        try:
            sample_df = conn.query(query, ttl=0) 
            if not sample_df.empty:
                st.session_state.text_id = sample_df.iloc[0]['id']
                st.session_state.text = sample_df.iloc[0]['companyactivity']
            else:
                st.warning("No new samples to review!")
                time.sleep(5)
                st.session_state.sample_fetched = False
                st.rerun()
        except Exception as e:
            st.error(f"Database Error! retry after 5 seconds.")
            time.sleep(5)
            st.session_state.sample_fetched = False
            st.rerun()
        
        # Run model inference
        with st.spinner("Running inference..."):
            st.session_state.predictions = predict_sorted(st.session_state.text, tokenizer, model, THRESHOLD_DEFAULT)

            # Set the flag to True, so we know to display the results.
            st.session_state.sample_fetched = True

    # run if the button was clicked OR if a checkbox was clicked.
    if st.session_state.sample_fetched:
        # Get data from session state
        text_id = str(st.session_state.text_id)
        text = st.session_state.text
        predictions = st.session_state.predictions
        
        # Display and Styling Logic (largely the same as your code) ---
        st.markdown(
            f"""
            <link href="https://cdn.fontcdn.ir/Font/Persian/Shabnam/Shabnam.css" rel="stylesheet" type="text/css">
            <div style="text-align: justify; direction: rtl; font-family: 'Shabnam', sans-serif;">
                {text}
            </div>
            <style>
                .stCheckbox {{
                    direction: rtl;
                    font-family: 'Shabnam', sans-serif;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Post-process and display results
        if not predictions:
            st.info("No labels passed the threshold.")
            # Allow fetching a new sample if prediction fails
            if st.button("Try another sample", use_container_width=True):
                st.session_state.sample_fetched = False
                st.rerun()
            
        else:
            # Get the group details
            _, _, _, results_df = compute_hierarchy_probs(predictions, False)

            st.header("Prediction Feedback")
            
            # Get unique Sections
            sections = results_df['Section'].unique().tolist()
            for section in sections:
                st.markdown(
                    f'<h4 style="direction: rtl; font-family: \'Shabnam\', sans-serif;">🗂️ بخش: {section_dict[section]}</h4>',
                    unsafe_allow_html=True
                )

                # Filter DataFrame for the current section
                section_df = results_df[results_df["Section"] == section]
                divisions = section_df['Division'].unique().tolist()
                for division in divisions:
                    # We wrap everything in a div with our "rtl-container" class
                    st.markdown(
                        f'<h5 style="direction: rtl; font-family: \'Shabnam\', sans-serif;">🗃️ حوزه: {division_dict[division]}</h5>',
                        unsafe_allow_html=True
                    )

                    # Filter DataFrame for the current division
                    division_df = section_df[section_df["Division"] == division]
                    
                    # A placeholder for all
                    col1, col2 = st.columns([11, 1])
                    with col2:
                        pass
                    with col1:
                        # Count how many columns are needed.
                        num_groups = len(division_df)
                        cols = st.columns(num_groups)

                        # Use zip to iterate through columns and data rows simultaneously.
                        for col, (_, row) in zip(cols[::-1], division_df.iterrows()):
                            with col:
                                group_name = row['Group']
                                group_prob = row['Prob']
                                
                                unique_key = f'{text_id}_{label_dict_inv[group_name]}'
                                
                                st.checkbox("&nbsp;صحیح است؟", key=unique_key)
                                
                                st.markdown(f"""
                                <div style="direction: rtl; font-family: \'Shabnam\', sans-serif;">
                                    <b>{group_dict[group_name]}</b><br>
                                </div>
                                """, unsafe_allow_html=True)
                    st.divider()
            
            st.info("Review the predictions and check the boxes for all correct labels. When you're done, press **✅ Submit Feedback**.")
            st.divider()

            # Ask for confirmation
            if st.button("✅ Submit Feedback", use_container_width=True):
                st.session_state.confirm_submit = True

            # Show confirmation prompt
            if st.session_state.get("confirm_submit", False):
                st.warning("Are you sure you want to submit this feedback?")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ Yes, Submit", key="confirm_yes", use_container_width=True):
                        feedback_data = {
                            text_id: {
                                "full_text": text,
                                "feedback": [],
                                "labels": [label_dict_inv[p[0]] for p in predictions],
                            }
                        }
                        # Iterate through the results that were actually shown to the user
                        for _, row in results_df.iterrows():
                            group_name = row['Group']
                            key = f"{text_id}_{label_dict_inv[group_name]}"
                            if key in st.session_state and st.session_state[key]:
                                feedback_data[text_id]["feedback"].append(label_dict_inv[group_name])

                        with st.spinner("Submitting feedback to database..."):
                            try:
                                with conn.session as session:
                                    # Insert feedback
                                    insert_query = sqlalchemy.text("""
                                        INSERT INTO dbo.feedback (id, full_text, predicted_labels, feedback_labels)
                                        VALUES (:id, :full_text, :predicted_labels, :feedback_labels)
                                        ON CONFLICT (id) DO UPDATE
                                        SET full_text = EXCLUDED.full_text,
                                            predicted_labels = EXCLUDED.predicted_labels,
                                            feedback_labels = EXCLUDED.feedback_labels;
                                    """)
                                    session.execute(insert_query, {
                                        "id": int(text_id),
                                        "full_text": feedback_data[text_id]["full_text"],
                                        "predicted_labels": feedback_data[text_id]["labels"],
                                        "feedback_labels": feedback_data[text_id]["feedback"]
                                    })

                                    # Update activity table
                                    update_query = sqlalchemy.text("""
                                        UPDATE dbo.activity
                                        SET is_reviewed = TRUE
                                        WHERE id = :id;
                                    """)
                                    session.execute(update_query, {"id": int(text_id)})

                                    session.commit()

                                st.success("Feedback submitted successfully!")
                            except Exception as e:
                                st.error(f"Database error: {e}")
                                time.sleep(5)
                            
                            # Reset everything after submission or error
                            st.session_state.sample_fetched = False
                            st.session_state.confirm_submit = False
                            st.rerun()

                with col2:
                    if st.button("❌ Cancel", key="confirm_no", use_container_width=True):
                        st.session_state.confirm_submit = False
                        st.rerun()

    else:
        # This block of code runs before the button is clicked.
        st.info("Press the button above to fetch a sample text and review its prediction.")
