from pathlib import Path
import streamlit as st
import helper
import settings

# Configure Streamlit page
st.set_page_config(
    page_title="Waste Classification",
)

st.empty()
st.sidebar.title("Waste Classification")

model_path = Path(settings.DETECTION_MODEL)

# Title and Instructions
st.title("Deep Learning Based Waste Segregation System")
st.write("Upload an image detecting waste types. Results will be displayed in the side.")

# Custom styles
st.markdown(
    """
    <style>
        /* Hide the default 200MB note below the file uploader */
        div[data-testid="stFileUploader"] > div > span {
            display: none !important;
        }

        .stRecyclable {
            background-color: rgba(233,192,78,255);
            padding: 1rem 0.75rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            margin-top: 0 !important;
            font-size:18px !important;
        }
        .stNonBiodegradable {
            background-color: rgba(94,128,173,255);
            padding: 1rem 0.75rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            margin-top: 0 !important;
            font-size:18px !important;
        }
        .stHazardous {
            background-color: rgba(194,84,85,255);
            padding: 1rem 0.75rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            margin-top: 0 !important;
            font-size:18px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load YOLO model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
else:
    helper.choose_input_and_classify(model)

