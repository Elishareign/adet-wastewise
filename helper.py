from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
import numpy as np
import os


def load_model(model_path):
    model = YOLO(model_path)
    return model


def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_biodegradable_items = set(detected_items) & set(settings.NON_BIODEGRADABLE)
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    return recyclable_items, non_biodegradable_items, hazardous_items


def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ").lower()


def _initialize_placeholders():
    if 'recyclable_placeholder' not in st.session_state:
        st.session_state['recyclable_placeholder'] = st.sidebar.empty()
    if 'non_biodegradable_placeholder' not in st.session_state:
        st.session_state['non_biodegradable_placeholder'] = st.sidebar.empty()
    if 'hazardous_placeholder' not in st.session_state:
        st.session_state['hazardous_placeholder'] = st.sidebar.empty()


def _clear_placeholders():
    st.session_state['recyclable_placeholder'].empty()
    st.session_state['non_biodegradable_placeholder'].empty()
    st.session_state['hazardous_placeholder'].empty()
    st.session_state['last_detection_time'] = 0


def _display_detected_frames(model, st_frame, image, image_name="uploaded_image.jpg"):
    image = cv2.resize(image, (640, int(640 * (9 / 16))))

    _initialize_placeholders()

    if 'unique_classes' not in st.session_state:
        st.session_state['unique_classes'] = set()
    if 'last_detection_time' not in st.session_state:
        st.session_state['last_detection_time'] = 0

    # Clear after 3 seconds
    if st.session_state['last_detection_time'] and time.time() - st.session_state['last_detection_time'] > 3:
        _clear_placeholders()

    res = model.predict(image, conf=0.5)
    names = model.names
    detected_items = set()

    for result in res:
        new_classes = set([names[int(c)].strip().lower().replace(" ", "_") for c in result.boxes.cls])
        st.session_state['unique_classes'] = new_classes
        detected_items.update(new_classes)

        recyclable_items, non_biodegradable_items, hazardous_items = classify_waste_type(detected_items)

        # Clear previous results
        _initialize_placeholders()
        st.session_state['recyclable_placeholder'].markdown('')
        st.session_state['non_biodegradable_placeholder'].markdown('')
        st.session_state['hazardous_placeholder'].markdown('')

        if recyclable_items:
            items_str = "\n- ".join(remove_dash_from_class_name(item) for item in recyclable_items)
            st.session_state['recyclable_placeholder'].markdown(
                f"<div class='stRecyclable'><strong>Recyclable items:</strong>\n\n- {items_str}"
                f"<br><br><em>{settings.ADVICE['recyclable']}</em></div>",
                unsafe_allow_html=True
            )

        if non_biodegradable_items:
            items_str = "\n- ".join(remove_dash_from_class_name(item) for item in non_biodegradable_items)
            st.session_state['non_biodegradable_placeholder'].markdown(
                f"<div class='stNonBiodegradable'><strong>Non-Biodegradable items:</strong>\n\n- {items_str}"
                f"<br><br><em>{settings.ADVICE['non_biodegradable']}</em></div>",
                unsafe_allow_html=True
            )

        if hazardous_items:
            items_str = "\n- ".join(remove_dash_from_class_name(item) for item in hazardous_items)
            st.session_state['hazardous_placeholder'].markdown(
                f"<div class='stHazardous'><strong>Hazardous items:</strong>\n\n- {items_str}"
                f"<br><br><em>{settings.ADVICE['hazardous']}</em></div>",
                unsafe_allow_html=True
            )

        st.session_state['last_detection_time'] = time.time()

    res_plotted = res[0].plot()
    st_frame.image(res_plotted, channels="BGR")


def choose_input_and_classify(model):
    st.subheader("Choose Input Method")
    option = st.radio("Select image source:", ("Upload Image", "Use Webcam"))

    st_frame = st.empty()

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            max_file_size = 5 * 1024 * 1024
            uploaded_file.seek(0, 2)
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)

            if file_size > max_file_size:
                st.error("File is too large. Please upload an image under 5MB.")
                return

            if uploaded_file.type.startswith("image"):
                image_name = os.path.basename(uploaded_file.name)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)

                _display_detected_frames(model, st_frame, image, image_name=image_name)

    elif option == "Use Webcam":
        if st.button('Start Webcam Detection'):
            try:
                vid_cap = cv2.VideoCapture(0)
                st_frame = st.empty()
                stop_button = st.button("Stop Webcam")

                while vid_cap.isOpened():
                    success, frame = vid_cap.read()
                    if not success:
                        st.error("Failed to read from webcam.")
                        break

                    _display_detected_frames(model, st_frame, frame, image_name="webcam_frame.jpg")

                    if stop_button:
                        break

                vid_cap.release()

            except Exception as e:
                st.error(f"Error accessing webcam: {e}")
