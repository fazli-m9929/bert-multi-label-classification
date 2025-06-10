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
        st.session_state.text = ""
        st.session_state.predictions = []
        
    # Load cached model/tokenizer
    tokenizer, model = load_resources()
    
    if st.button("🔄 Fetch a New Sample for Review", use_container_width=True):
        # The button's only job is to get data and save it to our "memory".
        st.session_state.text_id = 12345
        st.session_state.text = 'ارایه کلیه خدمات مهندسی کشاورزی شامل مشاوره اجرا نظارت بر اجرای انواع طرحهای ابیاری قطره ای بارانی تحت فشار و برقی کردن چاه های کشاورزی جهت بهبود عملیات زراعی در مزارع کشاورزی و باغات و تسطیح و اماده سازی و ابخیزداری و زهکشی کلیه زمینهای کشاورزی باغداری ایجاد فضای سبز گلخانه های فضای ازاد تهیه تولید تکثیر و پرورش گلهای اپارتمانی و انواع نهال تهیه تولید خرید فروش بسته بندی واردات صادرات انواع کودهای شیمیایی انواع بذر و نشا و کمپوست و سموم نباتی و ماشین الات کشاورزی و دامپروری ارایه خدمات مشاوره در زمینه بهبود عملیات زراعی و باغداری شناسایی و ردیابی افات زراعی و از بین بردن این افات بصورت علمی شرکت در مناقصه ها و مزایده های دولتی و شخصی اخذ نمایندگی از شرکتهای داخلی و خارجی انجام کلیه امور بازرگانی شامل واردات صادرات ترخیص حق العمل کاری کلیه کالاهای مجاز از کلیه گمرکات و بنادر کشور و شرکت در مناقصه ها و مزایده های دولتی و شخصی اخذ نمایندگی از شرکتهای داخلی و خارجی انجام کلیه عملیات اجرایی انواع طرحهای مخابراتی تاسیساتی ساختمانی راهسازی و گاز رسانی ابرسانی شهری و روستایی صنعتی شامل حفاری کابل کشی لوله گذاری کانال سازی خاکبرداری و جدول کشی انجام کلیه امور پروژه های ساختمانی مسکونی اسفالت کاری اسکلت سازی و محوطه سازی و تخریب و خاکبرداری خاکریزی و پی کنی دیوار چینی کلیه امور نقشه کشی (Gps – GPRS) کلیه امور اسنادی (خدمات اسنادی قبض انبار منطقه ویژه اصلاحیه مانیفست صورتحساب مجوز خروج کالا (بیجک) تسویه و صورت مجالس شناور صورت وضعیت خروج شناور'
        
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
                                    <small>احتمال: {group_prob:.2f}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    st.divider()
            
            st.info("Review the predictions and check the boxes for all correct labels. When you're done, press **✅ Submit Feedback**.")
            st.markdown("---") # Visual separator

            if st.button("✅ Submit Feedback", use_container_width=True):
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
                    # Reconstruct the key to look it up in session_state
                    key = f"{text_id}_{label_dict_inv[group_name]}"
                    if key in st.session_state:
                        if st.session_state[key]:
                            feedback_data[text_id]["feedback"].append(label_dict_inv[group_name])
                
                # In a real app, you would save `feedback_data` to your database here.
                with st.spinner("Submitting feedback..."):
                    st.success("Feedback submitted successfully!")
                    st.write("Storing feedbacks to database")
                    st.json(feedback_data)

                # Reset the state to go back to the starting screen
                time.sleep(10)
                st.session_state.sample_fetched = False
                st.rerun()

    else:
        # This block of code runs before the button is clicked.
        st.info("Press the button above to fetch a sample text and review its prediction.")
