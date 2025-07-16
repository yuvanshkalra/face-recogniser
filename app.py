import streamlit as st
import face_recognition
import os
import cv2
import numpy as np
from PIL import Image
import shutil # For deleting directories

# --- Configuration ---
KNOWN_FACES_DIR = 'known_faces' # Using a relative path for easier deployment

# Ensure the known_faces directory exists
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# --- Admin Credentials (for demonstration purposes) ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "adminpassword"

# --- Global Data Storage for Known Faces (now re-populated on every run) ---
# These will be loaded every time the script reruns, as persistence is removed.
# They are still stored in session_state to be accessible across functions within a single rerun.
if 'known_face_encodings' not in st.session_state:
    st.session_state.known_face_encodings = []
if 'known_face_names' not in st.session_state:
    st.session_state.known_face_names = []

# --- Function to Load Known Faces and Generate Encodings (ALWAYS re-encodes) ---
def load_known_faces_and_encodings():
    """
    Scans the KNOWN_FACES_DIR, encodes all faces, and populates
    st.session_state.known_face_encodings and st.session_state.known_face_names.
    This function now *always* re-encodes as persistence is removed.
    """
    st.write("--- Re-encoding Known Faces (No Persistence) ---")
    st.warning("Faces are re-encoded on every application rerun. This can be slow.")

    if not os.path.exists(KNOWN_FACES_DIR):
        st.error(f"Error: '{KNOWN_FACES_DIR}' directory not found. Please create it and add your known face images.")
        st.session_state.known_face_encodings = []
        st.session_state.known_face_names = []
        return

    temp_encodings = []
    temp_names = []
    
    # Use a Streamlit spinner for the encoding process
    with st.spinner("Scanning and encoding known faces... This happens on every rerun."):
        for name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, name)
            if os.path.isdir(person_dir):
                # st.write(f"Processing images for: **{name}**") # Too verbose for Streamlit
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(person_dir, filename)
                        try:
                            image = face_recognition.load_image_file(image_path)
                            face_locations = face_recognition.face_locations(image)
                            face_encodings = face_recognition.face_encodings(image, face_locations)

                            if face_encodings:
                                temp_encodings.append(face_encodings[0])
                                temp_names.append(name)
                            else:
                                st.warning(f"   - No face found in {filename} for {name}. Skipping.")
                        except Exception as e:
                            st.error(f"   - Error processing {filename} for {name}: {e}")

    st.session_state.known_face_encodings = temp_encodings
    st.session_state.known_face_names = temp_names

    st.info(f"Finished encoding process. Total known faces: {len(st.session_state.known_face_encodings)}")

    if not st.session_state.known_face_encodings:
        st.warning("\nWarning: No known faces loaded or encoded. Face recognition will only identify 'Unknown'.")

# --- Function to Detect Faces in an Image ---
def detect_faces_in_image(image_bytes):
    """
    Detects and recognizes faces in an image provided as bytes.
    Returns the image with bounding boxes and names drawn.
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode image
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # face_recognition expects RGB

    face_locations = face_recognition.face_locations(image_rgb)
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

    if not face_locations:
        st.warning("No faces found in the uploaded image.")
        return image_rgb # Return original image if no faces

    display_image = image_bgr.copy() # Use BGR copy for drawing with OpenCV

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        if st.session_state.known_face_encodings:
            matches = face_recognition.compare_faces(st.session_state.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(st.session_state.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = st.session_state.known_face_names[best_match_index]

        # Draw a box around the face (OpenCV uses BGR)
        cv2.rectangle(display_image, (left, top), (right, bottom), (0, 255, 0), 2) # Green box

        # Draw a label with the name below the face
        cv2.rectangle(display_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(display_image, name, (left + 6, bottom - 6), font, 0.7, (0, 0, 0), 1) # Black text

    return cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) # Convert back to RGB for Streamlit display

# --- App State Management ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_type' not in st.session_state:
    st.session_state.user_type = None # 'admin' or 'user'

# --- Login Functions ---
def admin_login_page():
    st.subheader("Admin Login")
    username = st.text_input("Username", key="admin_user")
    password = st.text_input("Password", type="password", key="admin_pass")

    if st.button("Login as Admin", key="admin_login_btn", help="Click to log in as Admin"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.user_type = 'admin'
            st.success("Admin logged in successfully!")
            st.experimental_rerun() # Rerun to switch to admin dashboard
        else:
            st.error("Invalid Admin credentials.")

def user_login_page():
    st.subheader("User Login")
    username = st.text_input("Username (any)", key="user_user")
    password = st.text_input("Password (any)", type="password", key="user_pass")

    if st.button("Login as User", key="user_login_btn", help="Click to log in as User"):
        if username and password: # Simple check for non-empty credentials
            st.session_state.logged_in = True
            st.session_state.user_type = 'user'
            st.success("User logged in successfully!")
            st.experimental_rerun() # Rerun to switch to user dashboard
        else:
            st.error("Please enter a username and password.")

# --- Admin Dashboard ---
def admin_dashboard():
    st.title("Admin Dashboard")
    st.write(f"Welcome, {st.session_state.user_type}!")

    st.markdown("---")
    st.subheader("Manage Known Faces Database")

    # Display current known faces
    st.write("### Current Known Faces:")
    if st.session_state.known_face_names:
        names_count = {}
        for name in st.session_state.known_face_names:
            names_count[name] = names_count.get(name, 0) + 1
        for name, count in names_count.items():
            st.write(f"- **{name}** (with {count} encodings)")
    else:
        st.info("No known faces in the database.")

    st.markdown("---")
    st.subheader("Add New Faces to Database")
    new_person_name = st.text_input("Enter name for new person/folder:", key="new_person_name_input")
    uploaded_files = st.file_uploader("Upload images for this person", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="admin_file_uploader")

    if new_person_name and uploaded_files:
        if st.button(f"Add {len(uploaded_files)} image(s) for {new_person_name}", key="add_images_btn"):
            person_path = os.path.join(KNOWN_FACES_DIR, new_person_name)
            if not os.path.exists(person_path):
                os.makedirs(person_path)
                st.info(f"Created directory: {person_path}")

            for uploaded_file in uploaded_files:
                file_path = os.path.join(person_path, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved: {uploaded_file.name} for {new_person_name}")

            # After adding, force re-encoding
            load_known_faces_and_encodings()
            st.rerun() # Rerun to update the displayed list of known faces

    st.markdown("---")
    st.subheader("Delete Faces from Database")
    person_to_delete = st.text_input("Enter name of person/folder to delete:", key="delete_person_name_input")
    if person_to_delete:
        delete_path = os.path.join(KNOWN_FACES_DIR, person_to_delete)
        if os.path.exists(delete_path) and os.path.isdir(delete_path):
            if st.button(f"Delete all images for {person_to_delete}", key="delete_person_btn", help="This will permanently delete the folder and its contents."):
                try:
                    shutil.rmtree(delete_path)
                    st.success(f"Successfully deleted folder for {person_to_delete}.")
                    # Force re-encoding after deletion
                    load_known_faces_and_encodings()
                    st.rerun() # Rerun to update the displayed list of known faces
                except Exception as e:
                    st.error(f"Error deleting {person_to_delete}: {e}")
        else:
            st.warning(f"Folder for '{person_to_delete}' not found.")

    # Removed "Rebuild Cache" button as it's no longer necessary with no persistence
    st.markdown("---")
    st.info("Note: With persistence removed, faces are re-encoded automatically on relevant actions (e.g., adding/deleting faces) and on every app rerun.")


    if st.button("Logout", key="admin_logout_btn"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.success("Logged out.")
        st.experimental_rerun()

# --- User Dashboard ---
def user_dashboard():
    st.title("User Dashboard")
    st.write(f"Welcome, {st.session_state.user_type}!")

    st.markdown("---")
    st.subheader("Recognize Faces from Uploaded Image")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="user_file_uploader")

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Recognize Faces in Uploaded Image", key="recognize_upload_btn"):
            with st.spinner("Recognizing faces..."):
                processed_image = detect_faces_in_image(uploaded_image.read())
                st.image(processed_image, caption="Faces Recognized", use_column_width=True)

    st.markdown("---")
    st.subheader("Recognize Faces from Webcam")
    st.info("Click 'Take Photo' to capture a single frame from your webcam for recognition.")
    camera_image = st.camera_input("Take a photo", key="user_camera_input")

    if camera_image:
        st.image(camera_image, caption="Captured Image", use_column_width=True)
        if st.button("Recognize Faces in Captured Image", key="recognize_camera_btn"):
            with st.spinner("Recognizing faces..."):
                processed_image = detect_faces_in_image(camera_image.read())
                st.image(processed_image, caption="Faces Recognized", use_column_width=True)

    if st.button("Logout", key="user_logout_btn"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.success("Logged out.")
        st.experimental_rerun()

# --- Main Application Layout ---
def main():
    # Load encodings every time the app runs, as there's no persistence
    load_known_faces_and_encodings()

    st.set_page_config(layout="centered", page_title="SSO Face Recognizer")

    # Custom CSS for red buttons and centering
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #FF4B4B; /* Red color */
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #CC0000; /* Darker red on hover */
            box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
        }
        .centered-title {
            text-align: center;
            font-size: 3em;
            color: #333;
            margin-top: 50px;
            margin-bottom: 20px;
        }
        .centered-subheader {
            text-align: center;
            font-size: 1.2em;
            color: #555;
            margin-bottom: 40px;
        }
        .stImage {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        /* Style for the main container to center content */
        .main .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding-top: 2rem; /* Adjust as needed */
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='centered-title'>SSO Face Recognizer</h1>", unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.markdown("<p class='centered-subheader'>Please choose your login type.</p>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            user_login_page()
        with col2:
            admin_login_page()
    else:
        if st.session_state.user_type == 'admin':
            admin_dashboard()
        elif st.session_state.user_type == 'user':
            user_dashboard()

if __name__ == "__main__":
    main()

