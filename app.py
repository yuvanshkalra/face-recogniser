import streamlit as st
import face_recognition
import cv2
import numpy as np
import json # For parsing Firebase credentials
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app

# --- Admin Credentials (loaded from Streamlit secrets) ---
ADMIN_USERNAME = st.secrets["admin_credentials"]["username"]
ADMIN_PASSWORD = st.secrets["admin_credentials"]["password"]

# --- Firebase Initialization ---
# Check if Firebase app is already initialized to prevent re-initialization errors
if not firebase_admin._apps:
    try:
        # Load Firebase credentials from Streamlit secrets
        firebase_credentials_json = json.loads(st.secrets["firebase_credentials"])
        cred = credentials.Certificate(firebase_credentials_json)
        initialize_app(cred)
        db = firestore.client() # Initialize Firestore client
        st.success("Firebase initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing Firebase: {e}. Make sure your Firebase credentials are correctly set in Streamlit secrets.")
        st.stop() # Stop the app if Firebase cannot be initialized
else:
    db = firestore.client() # Get existing Firestore client
    st.info("Firebase already initialized.")


# --- Global Data Storage for Known Faces ---
# These will be loaded from Firestore and stored in Streamlit's session state
if 'known_face_encodings' not in st.session_state:
    st.session_state.known_face_encodings = []
if 'known_face_names' not in st.session_state:
    st.session_state.known_face_names = []

# --- Function to Load Known Faces and Generate Encodings (from Firestore ONLY) ---
def load_known_faces_and_encodings():
    """
    Loads known face encodings and names ONLY from Firestore.
    """
    st.write("--- Loading Known Faces from Firestore ---")
    
    temp_encodings = []
    temp_names = []

    try:
        # Get all documents from the 'known_faces_data' collection
        docs = db.collection('known_faces_data').stream()

        for doc in docs:
            data = doc.to_dict()
            name = data.get('name')
            # Encodings are stored as lists in Firestore, convert back to numpy arrays
            encodings_list = data.get('encodings', [])
            
            if name and encodings_list:
                for enc_list in encodings_list:
                    temp_encodings.append(np.array(enc_list))
                    temp_names.append(name)
            else:
                st.warning(f"Skipping malformed document in Firestore: {doc.id}")

        st.session_state.known_face_encodings = temp_encodings
        st.session_state.known_face_names = temp_names
        st.success(f"Loaded {len(st.session_state.known_face_encodings)} known faces from Firestore.")

    except Exception as e:
        st.error(f"Error loading known faces from Firestore: {e}. Please check your Firebase setup and security rules.")
        st.session_state.known_face_encodings = []
        st.session_state.known_face_names = []

    if not st.session_state.known_face_encodings:
        st.info("No known faces in the Firestore database.")

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
    st.subheader("Manage Known Faces Database (Firestore as Source of Truth)")

    # Display current known faces from session state (which is loaded from Firestore)
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
    st.info("Uploaded images are processed directly to extract encodings, which are then stored in Firestore. Images are NOT saved locally.")
    new_person_name = st.text_input("Enter name for new person:", key="new_person_name_input")
    uploaded_files = st.file_uploader("Upload images for this person", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="admin_file_uploader")

    if new_person_name and uploaded_files:
        if st.button(f"Add {len(uploaded_files)} image(s) for {new_person_name}", key="add_images_btn"):
            new_encodings_for_person = []
            with st.spinner("Processing images and generating encodings..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Load image directly from bytes
                        image_bytes = uploaded_file.read()
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB) # Ensure RGB for face_recognition

                        face_locations = face_recognition.face_locations(image_rgb)
                        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

                        if face_encodings:
                            # Convert numpy array to list for Firestore storage
                            new_encodings_for_person.append(face_encodings[0].tolist())
                            st.info(f"Encoded face from {uploaded_file.name}.")
                        else:
                            st.warning(f"No face found in {uploaded_file.name}. Skipping encoding for this image.")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name} for encoding: {e}")

            if new_encodings_for_person:
                # Add/update document in Firestore
                doc_ref = db.collection('known_faces_data').document(new_person_name)
                
                # Fetch existing encodings if the person already exists
                existing_data = doc_ref.get()
                if existing_data.exists:
                    existing_encodings = existing_data.to_dict().get('encodings', [])
                    # Append new encodings to existing ones
                    all_encodings = existing_encodings + new_encodings_for_person
                    doc_ref.update({'encodings': all_encodings})
                    st.success(f"Updated encodings for {new_person_name} in Firestore.")
                else:
                    doc_ref.set({
                        'name': new_person_name,
                        'encodings': new_encodings_for_person
                    })
                    st.success(f"Added {new_person_name} and encodings to Firestore.")
            else:
                st.warning(f"No valid faces were encoded for {new_person_name} to add to Firestore.")

            # Reload known faces from Firestore to update the app's state
            load_known_faces_and_encodings()
            st.rerun() # Rerun to update the displayed list of known faces

    st.markdown("---")
    st.subheader("Delete Faces from Database")
    st.info("This will delete the person's data from Firestore. No local files are managed.")
    person_to_delete = st.text_input("Enter name of person to delete:", key="delete_person_name_input")
    if person_to_delete:
        if st.button(f"Delete {person_to_delete}", key="delete_person_btn", help="This will permanently delete the person's data."):
            # Delete from Firestore
            doc_ref = db.collection('known_faces_data').document(person_to_delete)
            if doc_ref.get().exists:
                doc_ref.delete()
                st.success(f"Successfully deleted {person_to_delete} from Firestore.")
            else:
                st.warning(f"Person '{person_to_delete}' not found in Firestore.")

            # Reload known faces from Firestore to update the app's state
            load_known_faces_and_encodings()
            st.rerun() # Rerun to update the displayed list of known faces

    st.markdown("---")
    st.info("All face data (encodings) is now stored and managed directly in Firestore. Changes are persistent across app restarts.")

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
    # Load encodings from Firestore when the app starts or on relevant actions
    # This ensures the app's state reflects the database content
    if not st.session_state.known_face_encodings: # Only load if not already loaded in this session
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

