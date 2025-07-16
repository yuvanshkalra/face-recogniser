import streamlit as st
import face_recognition
import os
import numpy as np
import cv2 # Still useful for BGR<->RGB conversion, resizing if not done by face_recognition

# --- Configuration (relative path for deployment) ---
# Ensure this directory is relative to your Streamlit app script and uploaded to GitHub
KNOWN_FACES_DIR = 'known_faces'

# --- Data Storage (use st.cache_resource for efficiency) ---
# @st.cache_resource caches the result of the function, so known faces are loaded only once
@st.cache_resource
def load_known_faces(known_faces_dir):
    st.info("Loading known faces... This might take a moment.")
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(known_faces_dir):
        st.error(f"Error: '{known_faces_dir}' directory not found on the server. "
                 "Please ensure it's included in your GitHub repository correctly.")
        return [], []

    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_locations = face_recognition.face_locations(image)
                        face_encodings = face_recognition.face_encodings(image, face_locations)

                        if face_encodings:
                            known_face_encodings.append(face_encodings[0])
                            known_face_names.append(name)
                        # Optional: Log to console for debugging during deployment
                        # else: print(f" - No face found in {filename} for {name}. Skipping.")
                    except Exception as e:
                        # Optional: Log to console for debugging during deployment
                        # print(f" - Error processing {filename} for {name}: {e}")
                        pass # Suppress detailed error for cleaner Streamlit output
    st.success(f"Finished loading known faces. Total known faces: {len(known_face_encodings)}")
    return known_face_encodings, known_face_names

# Load faces once when the app starts or is re-run due to cache invalidation
known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)

# --- Function to process an image and draw boxes (reusable) ---
def process_frame_for_faces(frame_rgb, known_encodings, known_names):
    """
    Detects and recognizes faces in an RGB image (NumPy array).
    Draws bounding boxes and names, then returns the BGR image.
    """
    # Ensure the input frame is writeable before drawing on it
    frame_rgb = np.copy(frame_rgb)
    
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR for drawing

    if not face_locations:
        return frame_bgr # Return original frame if no faces

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # --- Dynamic Box & Label Adjustments (same as your refined code) ---
        box_padding = 15
        base_label_height = 25
        text_y_offset = 10
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 1

        top_ext = max(0, top - box_padding)
        right_ext = min(frame_bgr.shape[1], right + box_padding)
        bottom_ext = min(frame_bgr.shape[0], bottom + box_padding)
        left_ext = max(0, left - box_padding)

        cv2.rectangle(frame_bgr, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), 2)

        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)
        label_width = text_width + (box_padding * 2) # Text width + padding on both sides
        label_height = max(base_label_height, text_height + (text_y_offset * 2)) # Min height or text height + padding

        label_top = bottom_ext # Start right below the face box
        label_bottom = label_top + label_height
        label_left = left_ext
        label_right = max(right_ext, left_ext + label_width)

        # Ensure label doesn't go below the image boundary
        if label_bottom > frame_bgr.shape[0]:
            label_bottom = frame_bgr.shape[0]
            label_top = label_bottom - label_height # Adjust top if bottom is clipped
            if label_top < 0: # Prevent label from going above image if height adjustment is too much
                label_top = 0

        # Ensure label doesn't go beyond right image boundary
        if label_right > frame_bgr.shape[1]:
            label_right = frame_bgr.shape[1]
            label_left = max(0, label_right - label_width) # Adjust left, but ensure it stays within bounds

        cv2.rectangle(frame_bgr, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED)

        text_x = label_left + (label_right - label_left - text_width) // 2
        text_y = label_top + (label_height + text_height) // 2 - baseline

        cv2.putText(frame_bgr, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame_bgr

# --- Streamlit UI ---
st.set_page_config(page_title="Dynamic Face Recognition App", layout="centered")
st.title("Face Recognition App with Dynamic Labels 🕵️‍♂️")

st.markdown("""
This application performs face recognition from your live webcam or an uploaded image.
The name labels will dynamically adjust their size to fit the recognized name!
""")

if not known_face_encodings:
    st.error("No known faces loaded. Please ensure your `known_faces` directory "
             "is correctly structured and contains images with faces for training.")

# Option selection
st.sidebar.header("Choose Input Method")
option = st.sidebar.radio("", ("Live Webcam Recognition", "Upload Image for Recognition"))

if option == "Live Webcam Recognition":
    st.subheader("Live Webcam Face Recognition")
    st.info("Allow camera access. Take a picture, and the app will detect/recognize faces.")

    # Use Streamlit's camera input
    camera_image = st.camera_input("Click 'Take Photo' to capture an image:", key="camera_input")

    if camera_image is not None:
        with st.spinner("Processing live image..."):
            # To read the image from BytesIO into numpy:
            file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # Read as BGR by default
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for face_recognition

            processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
            processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for st.image

        st.image(processed_img_rgb, caption="Processed Live Image", use_column_width=True)
        st.success("Face detection and recognition complete!")

    else:
        st.warning("Waiting for webcam input. Click 'Take Photo' above.")


elif option == "Upload Image for Recognition":
    st.subheader("Upload Image for Face Recognition")
    uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png", "bmp", "gif"], key="file_uploader")

    if uploaded_file is not None:
        with st.spinner("Loading and processing image..."):
            # To read the image from BytesIO into numpy:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # Read as BGR by default
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for face_recognition

            st.image(img_rgb, caption="Original Uploaded Image", use_column_width=True)

            processed_img_bgr = process_frame_for_faces(img_rgb, known_face_encodings, known_face_names)
            processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for st.image

        st.image(processed_img_rgb, caption="Processed Image with Faces", use_column_width=True)
        st.success("Face detection and recognition complete!")
    else:
        st.info("Please upload an image file using the browser button.")

st.markdown("---")
st.markdown("Developed with ❤️ using `face_recognition`, `OpenCV`, and `Streamlit`.")
