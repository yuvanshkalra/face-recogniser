
# 👤 Face Recogniser — Sports Profile Identification System

A Python-based face recognition web application built with Streamlit, OpenCV, and facial embeddings. The system identifies registered individuals from images, displays their associated profile information, and manages face records through Firebase.

## 🚀 Overview

Face Recogniser combines computer vision with cloud-based data management to provide an interactive face identification system.

Users can upload an image or capture a photo using their webcam to identify registered individuals. Recognized profiles display relevant information such as name, age, height, sport, and sport-specific statistics.

An administrator can register new faces, manage existing records, and update profile information through a dedicated admin panel.

## ✨ Key Features

- **Face Detection:** Detect faces in uploaded images and webcam captures.
- **Facial Recognition:** Extract facial encodings and compare them against registered faces.
- **Image-Based Identification:** Recognize individuals from JPG, JPEG, PNG, BMP, and GIF images.
- **User Authentication:** Separate user and administrator login flows.
- **Face Enrollment:** Register individuals by uploading a face image and entering profile information.
- **Admin Dashboard:** View, update, and manage registered face records.
- **Cloud Database:** Store face encodings and profile information in Firebase Cloud Firestore.
- **Cloud Image Storage:** Store registered face images in Firebase Storage.
- **Dynamic Profile Details:** Display recognized individuals' information based on their stored records.
- **Sports-Specific Profiles:** Support profile information for Cricket, NBA, and Football.
- **Interactive Interface:** Provide recognition results and profile details through Streamlit.

## 🧠 How It Works

The application follows a facial recognition pipeline:

1. **Input Acquisition:** Capture a photo using the webcam or upload an image.
2. **Face Detection:** Locate faces using the `face_recognition` library.
3. **Feature Extraction:** Convert detected faces into numerical facial encodings.
4. **Face Matching:** Compare the encodings against registered facial encodings stored in Firebase.
5. **Identity Recognition:** Use face comparison and face-distance calculations to find a matching identity.
6. **Profile Retrieval:** Retrieve the matched person's profile and associated sports information.
7. **Result Visualization:** Display recognized identities and their profile details in the Streamlit interface.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Web interface and authentication flows |
| face_recognition | Face detection, facial encodings, and face matching |
| OpenCV | Image decoding, color conversion, and image processing |
| NumPy | Numerical operations on images and encodings |
| Pillow (PIL) | Image loading and conversion |
| Firebase Admin SDK | Secure backend integration with Firebase |
| Cloud Firestore | Storage and management of face records |
| Firebase Storage | Storage of registered face images |
| CMake | Build support for native dependencies where required |

## 📁 Project Structure

```text
face-recogniser/
│
├── app.py
├── requirements.txt
├── README.md
└── sso_logo.jpg
```

### File Descriptions

- `app.py` — Main application containing the Streamlit interface, authentication, face recognition, and database management logic.
- `requirements.txt` — Python package dependencies.
- `sso_logo.jpg` — Logo displayed in the application sidebar.
- `README.md` — Project documentation.

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- A webcam or camera-enabled device for photo capture
- A Firebase project with Cloud Firestore and Firebase Storage configured

### 1. Clone the Repository

```bash
git clone https://github.com/yuvanshkalra/face-recogniser.git
cd face-recogniser
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project uses a Git-based installation of the `face_recognition` library. Its native dependencies may require CMake, a compatible dlib build environment, and additional system packages.

If installation fails, check the compatibility of your Python version and operating system with the installed packages.

## 🔥 Firebase Configuration

The application requires Firebase configuration to load known faces and manage profile records.

### Step 1: Create a Firebase Project

Visit the [Firebase Console](https://console.firebase.google.com/) and create or select a project.

### Step 2: Configure Firebase Services

- Enable Cloud Firestore.
- Set up Firebase Storage.
- Create the Firestore collection that will store registered face records.
- Configure an appropriate storage bucket.

### Step 3: Configure Streamlit Secrets

Create the following file locally:

`.streamlit/secrets.toml`

Use the following structure and replace the placeholders with your actual credentials and configuration:

```toml
[firebase]
service_account_json = '''PASTE_YOUR_COMPLETE_SERVICE_ACCOUNT_JSON_HERE'''
storage_bucket = "YOUR_FIREBASE_STORAGE_BUCKET"
firestore_collection = "YOUR_FIRESTORE_COLLECTION"

[admin]
username = "YOUR_ADMIN_USERNAME"
password = "YOUR_ADMIN_PASSWORD"

[users]
user1_username = "YOUR_USER_USERNAME"
user1_password = "YOUR_USER_PASSWORD"
```

**Important:** The service account value must contain valid JSON representing your complete Firebase service account credentials. The username and password fields are example placeholders; configure them according to your deployment requirements.

Never commit populated secrets files, Firebase private keys, or real passwords to GitHub. Use secure environment-specific configuration for deployment.

### Step 4: Prepare Face Records

Use the application's admin panel to register individuals and upload their images. The application generates facial encodings and saves the associated profile information in Firestore, with images stored in Firebase Storage.

## ▶️ Run the Application

Start the application using Streamlit:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## 🖥️ Using the Application

### User Workflow

1. Open the application.
2. Navigate to the user login page.
3. Sign in with configured user credentials.
4. Choose webcam capture or image upload.
5. Submit an image for recognition.
6. View recognized names and associated profile details.

### Administrator Workflow

1. Open the admin login page.
2. Sign in using the configured administrator credentials.
3. Open the Admin Panel.
4. Select **Add New Face** to register an individual.
5. Enter the person's name, sport, and relevant profile information.
6. Upload a face image.
7. Use **View/Update Database** to review records, edit details, or replace an image.

## 🏅 Sports Profile Management

The application supports profile information associated with three sports:

- **Cricket:** Total runs and total wickets.
- **NBA:** Points per game and player position.
- **Football:** Goals scored and goals saved.

Common profile fields include name, age, and height.

The available profile information depends on what has been entered into the database.

## 🔬 Technical Approach

### Facial Encoding

The application uses the `face_recognition` library to extract numerical representations of detected faces. These encodings are compared with stored encodings to identify potential matches.

### Face Comparison

The system uses face comparison and face-distance calculations to select a matching registered identity. Recognition reliability depends on the quality of the reference images and the matching threshold.

### Cloud Integration

Firebase Cloud Firestore stores structured face records and encodings, while Firebase Storage holds the associated images. This separates profile metadata from image files and allows records to be managed centrally.

## ⚠️ Limitations and Security Considerations

- Recognition accuracy can vary with lighting, image quality, camera angle, and facial occlusion.
- Unknown faces may be misidentified if matching thresholds are not appropriately configured.
- The webcam interface captures a photo for processing; it is not continuous video-stream recognition.
- Firebase connectivity and valid credentials are required for cloud-backed functionality.
- Basic username/password checks should not be treated as production-grade authentication without additional security controls.
- The system does not establish identity with certainty and should not be used as the sole basis for high-impact decisions.
- Obtain appropriate consent before collecting facial images and protect stored biometric data with access controls and retention policies.

## 🔮 Future Improvements

- Add liveness detection to help defend against photo-based spoofing.
- Improve authentication using a dedicated identity provider.
- Add stronger role-based access control.
- Introduce audit logs for administrative changes.
- Evaluate false-acceptance and false-rejection rates.
- Improve recognition performance across varied lighting and camera conditions.
- Add automated tests and deployment workflows.
- Introduce clearer biometric data deletion and retention controls.

## 👨‍💻 Author

**Yuvansh Kalra**

GitHub: [@yuvanshkalra](https://github.com/yuvanshkalra)

Repository: [Face Recogniser](https://github.com/yuvanshkalra/face-recogniser)

## 📄 License

Refer to the repository's license file for applicable usage and redistribution terms.

---

⭐ If you find this project useful, consider starring the repository.
