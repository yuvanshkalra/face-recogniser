import face_recognition # Library for face detection and recognition
import os # For interacting with the operating system
import cv2 # OpenCV library for image processing, video capture, and displaying images
import numpy as np # Numerical Python library, essential for array operations
import tkinter as tk # For basic GUI elements
from tkinter import filedialog # For file selection dialog

# --- Configuration ---
KNOWN_FACES_DIR = r'C:\Users\yuvan\OneDrive\Desktop\face-recognition-project\known_faces'

# --- Data Storage ---
known_face_encodings = []
known_face_names = []

# --- Part 1: Load Known Faces and Generate Encodings ---
print("--- Loading Known Faces ---")
print(f"Searching for known faces in: {KNOWN_FACES_DIR}")

if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Error: '{KNOWN_FACES_DIR}' directory not found. Please create it and add your known face images.")
    exit()
else:
    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_dir):
            print(f"Processing images for: {name}")
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
                            print(f"   - Added encoding for {name} from {filename}")
                        else:
                            print(f"   - No face found in {filename} for {name}. Skipping.")
                    except Exception as e:
                        print(f"   - Error processing {filename} for {name}: {e}")

    print(f"\nFinished loading known faces. Total known faces: {len(known_face_encodings)}")

# -----------------------------------------------------------------------------

# --- Function to open a file dialog and get image path ---
def ask_for_image_file():
    """
    Opens a file dialog to let the user select an image file.
    Returns the selected file path (str) or None if no file is selected.
    """
    # Create a Tkinter root window, but keep it hidden
    root = tk.Tk()
    root.withdraw()

    # Open the file dialog
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), # Common image formats
            ("All files", "*.*") # Option to show all file types
        ]
    )
    # Destroy the hidden root window after selection to clean up resources
    root.destroy()
    return file_path

# -----------------------------------------------------------------------------

# --- Function to detect faces in an uploaded image ---
def detect_faces_in_uploaded_image(image_path):
    """
    Detects and recognizes faces in a given image file.

    Args:
        image_path (str): The full path to the image file.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    print(f"\n--- Processing Uploaded Image: {os.path.basename(image_path)} ---")

    try:
        # Load the image using face_recognition (reads as RGB)
        uploaded_image_rgb = face_recognition.load_image_file(image_path)
        # Convert from RGB (face_recognition) to BGR (OpenCV for display)
        uploaded_image_bgr = cv2.cvtColor(uploaded_image_rgb, cv2.COLOR_RGB2BGR)

        # Find all faces in the uploaded image
        face_locations = face_recognition.face_locations(uploaded_image_rgb)
        face_encodings = face_recognition.face_encodings(uploaded_image_rgb, face_locations)

        if not face_locations:
            print("No faces found in the uploaded image.")
            # Display image even if no faces are found
            cv2.imshow('Uploaded Image - No Faces Found', uploaded_image_bgr)
            print("Press any key to close the image window.")
            cv2.waitKey(0) # Wait indefinitely until a key is pressed
            cv2.destroyAllWindows()
            return

        # Iterate through each face found in the current frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown" # Default name if no match is found

            # Only attempt to compare if there are known faces loaded.
            if known_face_encodings:
                # Compare the current unknown face's encoding with all known face encodings.
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                # Calculate the 'face distance' to each known face.
                # Lower distance means a closer match.
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # Find the index of the known face with the smallest distance (best match).
                best_match_index = np.argmin(face_distances)

                # If the best match is actually considered a 'match'
                if matches[best_match_index]:
                    name = known_face_names[best_match_index] # Assign the name of the matched person.

            # --- Adjustments for larger box and name ---
            # Define padding for the bounding box.
            box_padding = 15
            # Define a base height for the name label (will be adjusted dynamically)
            base_label_height = 25 # Minimum height for the text
            text_y_offset = 10 # Offset for text from top of label box

            # Font parameters
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            font_thickness = 1

            # Adjust coordinates for the face bounding box to make it slightly larger
            top_ext = max(0, top - box_padding)
            right_ext = min(uploaded_image_bgr.shape[1], right + box_padding)
            bottom_ext = min(uploaded_image_bgr.shape[0], bottom + box_padding)
            left_ext = max(0, left - box_padding)

            # Draw a box around the face (using extended coordinates)
            cv2.rectangle(uploaded_image_bgr, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), 2) # Green box

            # Calculate text size to dynamically adjust label width and height
            (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)

            # Determine dynamic label height and width
            label_width = text_width + (box_padding * 2) # Text width + padding on both sides
            label_height = max(base_label_height, text_height + (text_y_offset * 2)) # Min height or text height + padding

            # Calculate label box coordinates
            label_top = bottom_ext # Start right below the face box
            label_bottom = label_top + label_height
            # Label box should be at least as wide as the face box, or wider if name is long
            label_left = left_ext
            label_right = max(right_ext, left_ext + label_width)

            # Ensure label doesn't go below the image boundary
            if label_bottom > uploaded_image_bgr.shape[0]:
                label_bottom = uploaded_image_bgr.shape[0]
                label_top = label_bottom - label_height # Adjust top if bottom is clipped

            # Ensure label doesn't go beyond right image boundary
            if label_right > uploaded_image_bgr.shape[1]:
                label_right = uploaded_image_bgr.shape[1]
                # If clipped, try to adjust left, but prioritize keeping it attached to face
                label_left = max(0, label_right - label_width)


            # Draw a label with the name below the face
            cv2.rectangle(uploaded_image_bgr, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED)

            # Center the text horizontally within the (potentially adjusted) label box
            text_x = label_left + (label_right - label_left - text_width) // 2
            # Center the text vertically within the label box
            text_y = label_top + (label_height + text_height) // 2 - baseline

            cv2.putText(uploaded_image_bgr, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness) # Black text

        # Display the resulting image
        cv2.imshow(f'Uploaded Image - {os.path.basename(image_path)}', uploaded_image_bgr)
        print("Press any key to close the image window.")
        cv2.waitKey(0) # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing uploaded image '{image_path}': {e}")

# -----------------------------------------------------------------------------

# --- Main Application Logic (Interactive Menu) ---
while True:
    print("\n--- Choose an option ---")
    print("1. Start Real-time Face Recognition (Webcam)")
    print("2. Detect Faces in an Uploaded Image (Open File Dialog)")
    print("3. Exit")

    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == '1':
        print("\n--- Starting Real-time Face Recognition ---")
        print("Press 'q' to quit the video stream.")

        # Initialize webcam: 0 usually refers to the default webcam.
        video_capture = cv2.VideoCapture(0)

        # Check if the webcam was opened successfully
        if not video_capture.isOpened():
            print("Error: Could not open video stream. Make sure your webcam is connected and not in use.")
            continue # Go back to the main menu if webcam fails

        # Loop continuously to capture frames from the webcam
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # If frame was not read successfully, break the loop
            if not ret:
                print("Failed to grab frame. Exiting webcam stream...")
                break

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Iterate through each face found in the current frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown" # Default name if no match is found

                # Only attempt to compare if there are known faces loaded.
                if known_face_encodings:
                    # Compare the current unknown face's encoding with all known face encodings.
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    # Calculate the 'face distance' to each known face.
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                    # Find the index of the known face with the smallest distance (best match).
                    best_match_index = np.argmin(face_distances)

                    # If the best match is actually considered a 'match'
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index] # Assign the name of the matched person.

                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # --- Adjustments for larger box and name in webcam stream ---
                # Define padding for the bounding box.
                box_padding = 15
                # Define a base height for the name label (will be adjusted dynamically)
                base_label_height = 25 # Minimum height for the text
                text_y_offset = 10 # Offset for text from top of label box

                # Font parameters
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.7
                font_thickness = 1

                # Adjust coordinates for the face bounding box to make it slightly larger
                top_ext = max(0, top - box_padding)
                right_ext = min(frame.shape[1], right + box_padding)
                bottom_ext = min(frame.shape[0], bottom + box_padding)
                left_ext = max(0, left - box_padding)

                # Draw a box around the face (using extended coordinates)
                cv2.rectangle(frame, (left_ext, top_ext), (right_ext, bottom_ext), (0, 255, 0), 2) # Green box

                # Calculate text size to dynamically adjust label width and height
                (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)

                # Determine dynamic label height and width
                label_width = text_width + (box_padding * 2) # Text width + padding on both sides
                label_height = max(base_label_height, text_height + (text_y_offset * 2)) # Min height or text height + padding

                # Calculate label box coordinates
                label_top = bottom_ext # Start right below the face box
                label_bottom = label_top + label_height
                # Label box should be at least as wide as the face box, or wider if name is long
                label_left = left_ext
                label_right = max(right_ext, left_ext + label_width)

                # Ensure label doesn't go below the image boundary
                if label_bottom > frame.shape[0]:
                    label_bottom = frame.shape[0]
                    label_top = label_bottom - label_height # Adjust top if bottom is clipped

                # Ensure label doesn't go beyond right image boundary
                if label_right > frame.shape[1]:
                    label_right = frame.shape[1]
                    # If clipped, try to adjust left, but prioritize keeping it attached to face
                    label_left = max(0, label_right - label_width)

                # Draw a label with the name below the face
                cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), cv2.FILLED)

                # Center the text horizontally within the (potentially adjusted) label box
                text_x = label_left + (label_right - label_left - text_width) // 2
                # Center the text vertically within the label box
                text_y = label_top + (label_height + text_height) // 2 - baseline

                cv2.putText(frame, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness) # Black text

            # Display the resulting frame
            cv2.imshow('Video - Face Recognition (Press "q" to quit)', frame)

            # Wait for 1 millisecond and check if 'q' key was pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup after webcam stream
        video_capture.release()
        cv2.destroyAllWindows()
        print("--- Real-time Face Recognition Stopped ---")

    elif choice == '2':
        print("Opening file dialog... Please select an image file.")
        image_path_to_upload = ask_for_image_file()
        if image_path_to_upload:
            detect_faces_in_uploaded_image(image_path_to_upload)
        else:
            print("No image selected. Returning to main menu.")

    elif choice == '3':
        print("Exiting application. Goodbye! 👋")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

# Final cleanup in case of unexpected exit
cv2.destroyAllWindows()
