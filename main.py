# --- START OF FILE main.py ---

import os
import pickle
import base64
import io
import time
import logging
import json
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, UnidentifiedImageError
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly
import plotly.graph_objects as go

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(STATIC_FOLDER, 'files', 'models')
DATASET_DIR = os.path.join(STATIC_FOLDER, 'files', 'datasets')
DEFAULT_MODEL_NAME = 'logistic_regression_model.pkl' # Example default, change if needed
NO_HAND_TIMEOUT = 2.0 # Seconds before clearing sequence if no hand detected
EXPECTED_LANDMARK_FEATURES = 42 # 21 landmarks * 2 coordinates (x, y)

# --- Directory Setup ---
for dir_path in [MODEL_DIR, DATASET_DIR]:
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        except OSError as e:
            print(f"Error creating directory {dir_path}: {e}")
            exit(1) # Exit if essential dirs can't be made
    elif not os.path.isdir(dir_path):
        print(f"Error: {dir_path} exists but is not a directory.")
        exit(1)

# --- Initialize Flask, SocketIO, Logging ---
# Explicitly set static folder path for robustness
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_very_long_and_secure_dev_secret_key_CHANGE_ME_398dhq9_v3') # CHANGE THIS!
async_mode = None
# Use eventlet or gevent for better performance if installed
try:
    # import eventlet
    # eventlet.monkey_patch()
    async_mode = "eventlet"
    # print("Using eventlet for async mode.")
except ImportError:
    try:
        from gevent import monkey
        monkey.patch_all()
        async_mode = "gevent"
        print("Using gevent for async mode.")
    except ImportError:
        print("Warning: eventlet/gevent not found. Using default Flask server (lower performance).")

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
socketio_logger = logging.getLogger('socketio')
socketio_logger.setLevel(logging.INFO)
engineio_logger = logging.getLogger('engineio') # Separate logger for engineio
engineio_logger.setLevel(logging.INFO)
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.WARNING) # Quieter web server logs
app_logger = logging.getLogger(__name__) # Logger for application logic
app_logger.setLevel(logging.INFO)

# Initialize SocketIO
socketio = SocketIO(app, logger=socketio_logger, engineio_logger=engineio_logger)
user_sessions = {} # Dictionary to store session data per SID

# --- MediaPipe Initialization ---
hands_instance = None
mp_hands = None
mediapipe_init_error = None
try:
    mp_hands = mp.solutions.hands
    hands_instance = mp_hands.Hands(
        static_image_mode=False, # Process streams/sequences of images
        max_num_hands=1,         # Process only the most prominent hand
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    app_logger.info("MediaPipe Hands initialized successfully.")
except Exception as e:
    mediapipe_init_error = f"Failed to initialize MediaPipe Hands: {e}"
    app_logger.error(mediapipe_init_error) # Log the critical error

# --- Helper Functions ---

def get_session_data(sid):
    """Gets or initializes session data dictionary for a given SocketIO SID."""
    if sid not in user_sessions:
        app_logger.info(f"Initializing new session for SID: {sid}")
        user_sessions[sid] = {
            'model': None, 'model_name': None, 'sequence': [],
            'last_prediction': None, 'last_valid_prediction_time': time.time(),
            'last_activity_time': time.time(),
            'connect_time': time.time() # Add connection time for logging
        }
    else:
        user_sessions[sid]['last_activity_time'] = time.time() # Update activity time
    return user_sessions[sid]

def load_model(sid, model_name):
    """Loads the specified model file (.pkl) into the session data.
       Checks for expected structure and feature compatibility.
    """
    session_data = get_session_data(sid)
    path = os.path.join(MODEL_DIR, model_name)
    app_logger.info(f"Attempting to load model '{model_name}' for SID: {sid}")

    if not os.path.isfile(path):
        app_logger.warning(f"Model file '{model_name}' not found at path: {path}")
        return False, f"Model file '{model_name}' not found."

    try:
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        model_load_error = None # Track potential loading issues

        # --- Standardized Model Loading (Expects Dict) ---
        if isinstance(model_data, dict) and 'model' in model_data:
            actual_model = model_data.get('model')
            log_source = "dict key 'model'"

            # Validate the extracted model object
            if actual_model is None:
                model_load_error = "Model object within the dictionary is None."
            elif not hasattr(actual_model, 'predict'):
                model_load_error = "Loaded object does not have a 'predict' method."
            else:
                # Check feature compatibility if metadata exists
                loaded_features = model_data.get('expected_features')
                if loaded_features is not None and loaded_features != EXPECTED_LANDMARK_FEATURES:
                    compatibility_warning = (f"Model '{model_name}' was trained expecting {loaded_features} features, "
                                             f"but current configuration uses {EXPECTED_LANDMARK_FEATURES}. Predictions might be inaccurate.")
                    app_logger.warning(f"SID {sid}: {compatibility_warning}")
                    # Return success but include warning in the message
                    session_data['model'] = actual_model
                    session_data['model_name'] = model_name
                    app_logger.info(f"Successfully loaded model '{model_name}' from {log_source} for SID: {sid}, but with feature mismatch warning.")
                    return True, f"Model '{model_name}' loaded. Warning: {compatibility_warning}"
                else:
                    # Compatible or no feature info - Load normally
                    session_data['model'] = actual_model
                    session_data['model_name'] = model_name
                    app_logger.info(f"Successfully loaded model '{model_name}' (compatible features) from {log_source} for SID: {sid}")
                    return True, f"Model '{model_name}' loaded successfully."

        else:
             # --- Fallback for older/direct pickle files ---
             app_logger.warning(f"Model file '{model_name}' does not have the expected dict structure with a 'model' key. Attempting to load directly.")
             actual_model = model_data
             log_source = "pickle file directly"

             if actual_model is None:
                 model_load_error = "Directly loaded pickle data is None."
             elif not hasattr(actual_model, 'predict'):
                 model_load_error = f"Directly loaded object from '{model_name}' does not have a 'predict' method."
             else:
                 # Loaded directly, cannot verify features from metadata
                 session_data['model'] = actual_model
                 session_data['model_name'] = model_name
                 app_logger.info(f"Successfully loaded model '{model_name}' directly for SID: {sid} (feature compatibility not verified).")
                 return True, f"Model '{model_name}' loaded (compatibility not verified)."

        # --- Handle Errors During Loading ---
        if model_load_error:
            raise ValueError(model_load_error) # Raise error to be caught below

    except (pickle.UnpicklingError, AttributeError, EOFError, ValueError, TypeError) as e:
        error_msg = f"Error loading model '{model_name}': Invalid format, structure, or data - {e}"
        app_logger.error(error_msg, exc_info=True) # Log with traceback for these errors
        session_data['model'], session_data['model_name'] = None, None
        return False, f"Error loading model '{model_name}': Invalid file format or structure."
    except FileNotFoundError: # Should be caught earlier, but included for completeness
         app_logger.error(f"Model file not found error during loading: {path}")
         return False, f"Model file '{model_name}' not found."
    except Exception as e:
        app_logger.error(f"Unexpected error loading model '{model_name}': {e}", exc_info=True) # Log traceback
        session_data['model'], session_data['model_name'] = None, None
        return False, f"Unexpected error loading model '{model_name}'."


def base64_to_cv2_image(base64_string):
    """Safely decodes a base64 string (with/without prefix) to an OpenCV image (BGR format). Handles transparency."""
    if not isinstance(base64_string, str):
        app_logger.warning("Invalid input type for base64 decoding (expected string).")
        return None
    try:
        # Remove prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]

        # Decode base64
        img_data = base64.b64decode(base64_string)
        if not img_data:
            app_logger.warning("Empty image data after base64 decoding.")
            return None

        # Use Pillow (PIL) for robust opening and format handling
        pil_image = Image.open(io.BytesIO(img_data))

        # Handle transparency: convert RGBA/LA to RGB by pasting on a white background
        if pil_image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            alpha_channel = pil_image.split()[-1]
            background.paste(pil_image, mask=alpha_channel)
            pil_image = background

        # Convert other modes to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert Pillow RGB image to NumPy array
        img_array = np.array(pil_image)

        # Convert RGB NumPy array to OpenCV BGR format
        cv2_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        return cv2_image

    except base64.binascii.Error as e:
        app_logger.warning(f"Base64 decoding error: {e}")
        return None
    except UnidentifiedImageError as e:
        app_logger.warning(f"PIL could not identify image format: {e}")
        return None
    except Exception as e:
        app_logger.error(f"Unexpected error converting base64 to cv2 image: {e}", exc_info=True)
        return None

def _process_image_get_landmarks(sid, image_b64, log_prefix=""):
    """ Decodes base64 image, runs MediaPipe hands, extracts landmarks.
        Returns (1, N) numpy array of features or None.
    """
    if hands_instance is None:
        app_logger.warning(f"{log_prefix} SID {sid}: Landmark processing skipped, MediaPipe not initialized.")
        return None

    frame = base64_to_cv2_image(image_b64)
    if frame is None:
        return None

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands_instance.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
            if len(landmarks) == EXPECTED_LANDMARK_FEATURES:
                return np.array(landmarks, dtype=np.float32).reshape(1, -1)
            else:
                 app_logger.warning(f"{log_prefix} SID {sid}: Found hand, landmark count mismatch: Got {len(landmarks)}, expected {EXPECTED_LANDMARK_FEATURES}.")
                 return None
        else:
            return None
    except Exception as e:
        app_logger.error(f"{log_prefix} SID {sid}: Exception during MediaPipe/landmark extraction: {e}", exc_info=True)
        return None

def predict_from_landmarks(sid, landmarks_array):
    """Makes a prediction using the session's loaded model.
       Returns prediction string or specific error string.
    """
    session_data = get_session_data(sid)
    model = session_data.get('model')
    model_name = session_data.get('model_name', 'None')

    if model is None:
        # app_logger.warning(f"SID {sid}: Prediction attempted but no model loaded.") # Can be noisy
        return "No Model"

    if not isinstance(landmarks_array, np.ndarray) or landmarks_array.ndim != 2 or landmarks_array.shape[0] != 1 or landmarks_array.shape[1] != EXPECTED_LANDMARK_FEATURES:
        input_shape = landmarks_array.shape if isinstance(landmarks_array, np.ndarray) else type(landmarks_array)
        input_features = landmarks_array.shape[1] if isinstance(landmarks_array, np.ndarray) and landmarks_array.ndim == 2 else 'N/A'
        app_logger.warning(f"SID {sid}: Invalid landmark input for prediction. Expected shape (1, {EXPECTED_LANDMARK_FEATURES}), got {input_shape} with {input_features} features.")
        return "Input Error"

    try:
        prediction = model.predict(landmarks_array)
        predicted_char = str(prediction[0]).upper()
        return predicted_char
    except Exception as e:
        app_logger.error(f"SID {sid}: Error during prediction with model '{model_name}': {e}", exc_info=True)
        return "Predict Error"

def get_available_items(directory, extension):
    """Helper to get a sorted list of files with a specific extension from a directory."""
    items = []
    if not os.path.isdir(directory):
        app_logger.error(f"Directory for available items not found or invalid: {directory}")
        return []
    try:
        for f in os.listdir(directory):
            if f.endswith(extension) and os.path.isfile(os.path.join(directory, f)):
                items.append(f)
        return sorted(items)
    except Exception as e:
        app_logger.error(f"Error listing items in {directory} with extension {extension}: {e}", exc_info=True)
        return []

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main landing page (`index.html`)."""
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/RTP')
def rtp():
    """Serves the Real-Time Processing page (`real_time_processing.html`)."""
    if mediapipe_init_error:
         app_logger.error(f"Access to /RTP denied due to MediaPipe error: {mediapipe_init_error}")
         return f"Server Error: Real-time processing is unavailable because MediaPipe failed to initialize. Details: {mediapipe_init_error}", 500
    model_files = get_available_items(MODEL_DIR, '.pkl')
    return render_template('real_time_processing.html', model_files=model_files)

@app.route('/ImageProcessing')
def image_processing():
    """Serves the Image Upload Processing page (`image_processing.html`)."""
    if mediapipe_init_error:
        app_logger.error(f"Access to /ImageProcessing denied due to MediaPipe error: {mediapipe_init_error}")
        return f"Server Error: Image processing is unavailable because MediaPipe failed to initialize. Details: {mediapipe_init_error}", 500
    model_files = get_available_items(MODEL_DIR, '.pkl')
    return render_template('image_processing.html', model_files=model_files)

@app.route('/train')
def train_page():
    """Serves the Model Training page (`train.html`)."""
    datasets = get_available_items(DATASET_DIR, '.pkl')
    algorithms = list(ALGORITHM_IMPLEMENTATIONS.keys())
    try:
        relative_dataset_dir = os.path.relpath(DATASET_DIR, BASE_DIR)
    except ValueError:
        relative_dataset_dir = DATASET_DIR
    return render_template('train.html', datasets=datasets, algorithms=algorithms, DATASET_DIR=relative_dataset_dir)

@app.route('/compare')
def compare_page():
    """Serves the Model Comparison page (`compare_models.html`)."""
    datasets = get_available_items(DATASET_DIR, '.pkl')
    # Map keys to user-friendly names for the comparison dropdown
    algorithms_display_names = {
        'random forest model': 'Random Forest',
        'logistic regression model': 'Logistic Regression',
        'svm model': 'SVM',
        'naive bayes model': 'Naive Bayes',
        'decision tree model': 'Decision Tree',
        'lda model': 'LDA'
    }
    available_comparison_algos = {k: algorithms_display_names.get(k, k.title())
                                   for k in ALGORITHM_IMPLEMENTATIONS.keys()}

    try:
        relative_dataset_dir = os.path.relpath(DATASET_DIR, BASE_DIR)
    except ValueError:
        relative_dataset_dir = DATASET_DIR
    print('Navigating to compare_models.html')
    return render_template('compare_models.html',
                           datasets=datasets,
                           algorithms=available_comparison_algos,
                           DATASET_DIR=relative_dataset_dir)

# --- Machine Learning Model Implementations ---
ALGORITHM_IMPLEMENTATIONS = {
    'random forest model': (RandomForestClassifier, {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}),
    'logistic regression model': (LogisticRegression, {'max_iter': 2000, 'random_state': 42, 'solver': 'liblinear', 'C': 1.0}),
    'svm model': (SVC, {'probability': True, 'random_state': 42, 'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}),
    'naive bayes model': (GaussianNB, {}),
    'decision tree model': (DecisionTreeClassifier, {'random_state': 42, 'max_depth': None, 'min_samples_split': 2}),
    'lda model': (LinearDiscriminantAnalysis, {'solver': 'svd'})
}

# --- Data Loading Function ---
def load_data(file_path):
    """Load dataset from a pickle file. Returns tuple: (features_array, labels_array).
       Validates structure and feature dimensions.
    """
    app_logger.info(f"Loading dataset from: {file_path}")
    try:
        if not os.path.isfile(file_path):
             raise FileNotFoundError(f"Dataset file not found at {file_path}")

        with open(file_path, 'rb') as file:
             data_dict = pickle.load(file)

        if not isinstance(data_dict, dict) or 'data' not in data_dict or 'label' not in data_dict:
             raise ValueError("Dataset pickle must be a dict with 'data' and 'label' keys.")

        data_samples = data_dict['data']
        labels = data_dict['label']

        if not isinstance(data_samples, list) or not isinstance(labels, list):
             raise ValueError("'data' and 'label' must be lists.")
        if len(data_samples) != len(labels):
            raise ValueError(f"Data ({len(data_samples)}) and labels ({len(labels)}) have different lengths.")
        if not data_samples:
            app_logger.warning(f"Dataset file '{os.path.basename(file_path)}' is empty.")
            return np.empty((0, EXPECTED_LANDMARK_FEATURES), dtype=np.float32), np.empty((0,), dtype=object)

        try:
            first_sample_flat = np.asarray(data_samples[0], dtype=np.float32).flatten()
            dataset_feature_dim = len(first_sample_flat)

            if dataset_feature_dim != EXPECTED_LANDMARK_FEATURES:
                # **Important:** Keep this check stringent for training, maybe relax/warn for comparison?
                # For now, strict check applies to both /train and /compare load_data calls.
                 raise ValueError(f"Dataset samples have {dataset_feature_dim} features, expected {EXPECTED_LANDMARK_FEATURES}.")

            data_features = np.array([np.asarray(sample, dtype=np.float32).flatten() for sample in data_samples])

        except (ValueError, TypeError, IndexError) as e:
            raise ValueError(f"Error processing dataset features: {e}. Check sample structure consistency ({EXPECTED_LANDMARK_FEATURES} landmarks).")

        labels_array = np.array(labels)

        app_logger.info(f"Dataset loaded: {data_features.shape[0]} samples, {data_features.shape[1]} features.")
        return data_features, labels_array

    except FileNotFoundError as e:
        app_logger.error(f"Dataset file not found: {file_path}")
        raise
    except (pickle.UnpicklingError, ValueError, TypeError, IndexError) as e:
        app_logger.error(f"Error loading/parsing dataset {file_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to load/parse dataset '{os.path.basename(file_path)}'. Check format ('data'/'label' keys, correct data types).") from e
    except Exception as e:
        app_logger.error(f"Unexpected error loading dataset {file_path}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error loading dataset '{os.path.basename(file_path)}'.") from e

# --- Model Saving Function ---
def save_model(model_object, model_save_path, model_metadata):
    """Saves the trained model object along with provided metadata dictionary."""
    app_logger.info(f"Saving model to: {model_save_path}")
    try:
        dict_to_save = model_metadata.copy()
        dict_to_save['model'] = model_object
        dict_to_save['save_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        dict_to_save['expected_features'] = EXPECTED_LANDMARK_FEATURES # Store expectation

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, 'wb') as file:
             pickle.dump(dict_to_save, file, protocol=pickle.HIGHEST_PROTOCOL)
        app_logger.info(f"Model and metadata saved successfully to: {model_save_path}")
    except Exception as e:
        app_logger.error(f"Failed to save model to {model_save_path}: {e}", exc_info=True)
        raise IOError(f"Could not save model to '{os.path.basename(model_save_path)}'. Check permissions.")

# --- Core Training and Evaluation Function (Used by /train and /compare) ---
def train_model(data, labels, model_instance, test_size=0.2):
    """Trains the provided model instance, evaluates, and returns performance metrics."""
    model_name = type(model_instance).__name__
    app_logger.info(f"Starting train/eval pipeline for {model_name} ({data.shape[0]} samples)...")

    if not isinstance(data, np.ndarray) or not isinstance(labels, np.ndarray):
         raise TypeError("Input data and labels must be NumPy arrays.")
    if data.shape[0] == 0 or labels.shape[0] == 0:
        raise ValueError("Cannot train model: Input data or labels are empty.")
    if data.shape[0] != labels.shape[0]:
        raise ValueError(f"Data ({data.shape[0]}) and labels ({labels.shape[0]}) sample count mismatch.")
    if data.shape[1] != EXPECTED_LANDMARK_FEATURES:
         raise ValueError(f"Input data feature mismatch: Expected {EXPECTED_LANDMARK_FEATURES}, got {data.shape[1]}.")

    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples_per_class = np.min(counts) if len(counts) > 0 else 0
    if min_samples_per_class < 2 and len(unique_labels) > 0:
         raise ValueError(f"Cannot perform stratified split: Class '{unique_labels[np.argmin(counts)]}' has only {min_samples_per_class} sample(s). Need at least 2.")

    required_test_samples_per_class = 1 # Need at least 1 sample per class in the test set for stratification to work properly
    required_train_samples_per_class = 1 # Need at least 1 sample per class in the train set

    # Calculate the minimum number of samples required per class for the given test size
    # A class needs at least 'required_test_samples_per_class / test_size' samples total for split
    min_total_samples_for_split = max(required_train_samples_per_class + required_test_samples_per_class,
                                      int(np.ceil(required_test_samples_per_class / test_size)))

    if min_samples_per_class < min_total_samples_for_split and len(unique_labels) > 0:
        app_logger.warning(f"Low sample count ({min_samples_per_class}) for class '{unique_labels[np.argmin(counts)]}'. Need {min_total_samples_for_split} for robust stratified split with test_size={test_size}. Using all data for training.")
        # Fallback: Train on all data, return None for accuracy as test set is unreliable/empty
        try:
            start_time = time.time()
            app_logger.info(f"Training {model_name} on all {data.shape[0]} samples due to small class size...")
            model_instance.fit(data, labels)
            training_duration = time.time() - start_time
            app_logger.info(f"Full dataset training complete for {model_name}. Duration: {training_duration:.3f}s.")
            # Cannot calculate meaningful accuracy here
            return model_instance, None, training_duration, data.shape[0], 0 # Accuracy is None, test_count is 0
        except Exception as e:
             app_logger.error(f"Error during {model_name} full dataset training: {e}", exc_info=True)
             raise RuntimeError(f"Model training failed for {model_name} (full dataset): {e}") from e

    try:
        start_time = time.time()
        app_logger.info(f"Splitting {data.shape[0]} samples (test_size={test_size})...")
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels,
            test_size=test_size,
            shuffle=True,
            stratify=labels,
            random_state=42
        )
        train_count = x_train.shape[0]
        test_count = x_test.shape[0]
        if train_count == 0 or test_count == 0:
             raise RuntimeError("Empty train or test set after splitting (should have been caught by sample check).")

        app_logger.info(f"Training {model_name} with {train_count} samples...")
        model_instance.fit(x_train, y_train)
        training_duration = time.time() - start_time

        app_logger.info(f"Evaluating {model_name} on {test_count} test samples...")
        y_pred = model_instance.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        app_logger.info(f"Train/Eval complete for {model_name}. Duration: {training_duration:.3f}s, Accuracy: {accuracy:.4f}")
        return model_instance, accuracy, training_duration, train_count, test_count

    except ValueError as ve:
        app_logger.error(f"ValueError during {model_name} train/eval: {ve}", exc_info=True)
        raise RuntimeError(f"Model training process failed for {model_name} due to data issue: {ve}") from ve
    except Exception as e:
        app_logger.error(f"Unexpected error during {model_name} train/eval: {e}", exc_info=True)
        raise RuntimeError(f"Model training process failed unexpectedly for {model_name}: {e}") from e

# --- Training Route Logic ---

# Wrapper function for training AND saving (used only by /train_model route)
def _train_and_save_wrapper(model_class, model_params, model_metadata, data, labels, model_path):
    """Helper to instantiate, train, save a model, and return performance metrics."""
    app_logger.info(f"Training & Saving: {model_metadata['model_name']} to {os.path.basename(model_path)}...")
    try:
        model_instance = model_class(**model_params)
        trained_model, accuracy, duration, train_count, test_count = train_model(
            data, labels, model_instance
        )

        # Handle case where accuracy is None (full dataset training)
        if accuracy is None:
            app_logger.warning(f"Model {model_metadata['model_name']} trained on full dataset due to small class size. Accuracy cannot be reported reliably.")
            accuracy_value = None # Represent missing accuracy as None
        else:
            accuracy_value = accuracy

        # Add metadata *before* saving
        model_metadata['accuracy'] = accuracy_value # Store None if not calculable
        model_metadata['training_duration_sec'] = duration
        model_metadata['train_sample_count'] = train_count
        model_metadata['test_sample_count'] = test_count
        model_metadata['dataset_used'] = model_metadata.get('dataset_filename', 'Unknown')
        model_metadata['training_timestamp'] = time.time() # Add timestamp of training completion

        save_model(trained_model, model_path, model_metadata)
        app_logger.info(f"Saved {model_metadata['model_name']} to {os.path.basename(model_path)}. Acc: {accuracy_value}, Time: {duration:.3f}s")
        return accuracy_value, duration, train_count, test_count

    except (ValueError, RuntimeError, TypeError, IOError) as e:
         model_name = model_metadata.get('model_name', 'Unknown Model')
         app_logger.error(f"Error in training/saving wrapper for {model_name}: {e}", exc_info=True)
         raise e
    except Exception as e:
         model_name = model_metadata.get('model_name', 'Unknown Model')
         app_logger.error(f"Unexpected error in wrapper for {model_name}: {e}", exc_info=True)
         raise RuntimeError(f"Unexpected failure during training/saving process for {model_name}.") from e

# Specific functions for each algorithm key (boilerplate calling the wrapper)
def random_forest_model_train_save(data, labels, model_path, dataset_filename):
    algo_key = 'random forest model'
    model_class, params = ALGORITHM_IMPLEMENTATIONS[algo_key]
    metadata = {'model_name': 'Random Forest Classifier', 'algorithm_key': algo_key, 'hyperparameters': params, 'dataset_filename': dataset_filename}
    return _train_and_save_wrapper(model_class, params, metadata, data, labels, model_path)

def logistic_regression_model_train_save(data, labels, model_path, dataset_filename):
    algo_key = 'logistic regression model'
    model_class, params = ALGORITHM_IMPLEMENTATIONS[algo_key]
    metadata = {'model_name': 'Logistic Regression', 'algorithm_key': algo_key, 'hyperparameters': params, 'dataset_filename': dataset_filename}
    return _train_and_save_wrapper(model_class, params, metadata, data, labels, model_path)

def svm_model_train_save(data, labels, model_path, dataset_filename):
    algo_key = 'svm model'
    model_class, params = ALGORITHM_IMPLEMENTATIONS[algo_key]
    metadata = {'model_name': 'Support Vector Machine (SVM)', 'algorithm_key': algo_key, 'hyperparameters': params, 'dataset_filename': dataset_filename}
    return _train_and_save_wrapper(model_class, params, metadata, data, labels, model_path)

def naive_bayes_model_train_save(data, labels, model_path, dataset_filename):
    algo_key = 'naive bayes model'
    model_class, params = ALGORITHM_IMPLEMENTATIONS[algo_key]
    metadata = {'model_name': 'Gaussian Naive Bayes', 'algorithm_key': algo_key, 'hyperparameters': params, 'dataset_filename': dataset_filename}
    return _train_and_save_wrapper(model_class, params, metadata, data, labels, model_path)

def decision_tree_model_train_save(data, labels, model_path, dataset_filename):
    algo_key = 'decision tree model'
    model_class, params = ALGORITHM_IMPLEMENTATIONS[algo_key]
    metadata = {'model_name': 'Decision Tree Classifier', 'algorithm_key': algo_key, 'hyperparameters': params, 'dataset_filename': dataset_filename}
    return _train_and_save_wrapper(model_class, params, metadata, data, labels, model_path)

def lda_model_train_save(data, labels, model_path, dataset_filename):
    algo_key = 'lda model'
    model_class, params = ALGORITHM_IMPLEMENTATIONS[algo_key]
    metadata = {'model_name': 'Linear Discriminant Analysis (LDA)', 'algorithm_key': algo_key, 'hyperparameters': params, 'dataset_filename': dataset_filename}
    return _train_and_save_wrapper(model_class, params, metadata, data, labels, model_path)

# Route to Handle Training Request from train.html
@app.route('/train_model', methods=['POST'])
def handle_train_model():
    """Handles AJAX request to train and save a model, generates Plotly JSON for charts."""
    app_logger.info("Received POST request to /train_model")
    if not request.is_json:
        app_logger.warning("Training request rejected: Not JSON.")
        return jsonify({'message': 'Request content type must be application/json.'}), 415

    data = request.get_json()
    dataset_filename = data.get('dataset')
    algorithm_key = data.get('algorithm')
    new_model_name_base = data.get('model-name')

    # --- Input Validation ---
    errors = {}
    if not dataset_filename: errors['dataset'] = 'Dataset required.'
    if not algorithm_key: errors['algorithm'] = 'Algorithm required.'
    if not new_model_name_base:
        errors['model-name'] = 'Model name required.'
    else:
        sanitized_model_name = "".join(c for c in new_model_name_base if c.isalnum() or c in ['_', '-']).strip()
        if not sanitized_model_name:
            errors['model-name'] = 'Invalid chars in model name (use letters, numbers, _, -).'
        elif len(sanitized_model_name) > 50:
             errors['model-name'] = 'Model name too long (max 50).'
        else:
            new_model_name_base = sanitized_model_name

    if errors:
        app_logger.warning(f"Training validation failed: {errors}")
        return jsonify({'message': "Validation failed: " + "; ".join(errors.values())}), 400

    # Construct final paths
    model_filename = f"{new_model_name_base}.pkl"
    dataset_path = os.path.join(DATASET_DIR, dataset_filename)
    model_path = os.path.join(MODEL_DIR, model_filename)

    if os.path.exists(model_path):
        app_logger.warning(f"Training aborted: Model file '{model_filename}' already exists.")
        return jsonify({'message': f"Model file '{model_filename}' already exists. Choose a different name."}), 409 # Conflict

    app_logger.info(f"Training: Dataset='{dataset_filename}', Algo='{algorithm_key}', Output='{model_filename}'")

    try:
        data_features, labels = load_data(dataset_path)
        training_functions = {
            'random forest model': random_forest_model_train_save,
            'logistic regression model': logistic_regression_model_train_save,
            'svm model': svm_model_train_save,
            'naive bayes model': naive_bayes_model_train_save,
            'decision tree model': decision_tree_model_train_save,
            'lda model': lda_model_train_save
        }
        if algorithm_key not in training_functions:
            raise ValueError(f"Invalid algorithm specified: {algorithm_key}")

        train_func = training_functions[algorithm_key]
        accuracy, duration, train_count, test_count = train_func(data_features, labels, model_path, dataset_filename)

        dist_plot_json = generate_label_distribution_plotly_json(labels, dataset_filename)
        # Handle case where split didn't happen (test_count=0) for split plot
        split_plot_json = generate_data_split_plotly_json(train_count, test_count, dataset_filename) if test_count > 0 else None

        if accuracy is None: # Trained on full dataset
            accuracy_display = "N/A (Trained on full dataset)"
        else:
            accuracy_display = f"{accuracy*100:.2f}%"

        success_message = f"Model '{model_filename}' trained successfully using '{algorithm_key}'. Accuracy: {accuracy_display}"
        app_logger.info(success_message + f" | Time: {duration:.3f}s | Train/Test: {train_count}/{test_count}")

        response_data = {
            'message': success_message,
            'accuracy': accuracy, # Send the raw value (or None) for JS
            'training_time': duration,
            'distribution_plot_json': dist_plot_json,
            'data_split_plot_json': split_plot_json
        }
        return jsonify(response_data), 200

    except FileNotFoundError as e:
        err_msg = f"Dataset file '{dataset_filename}' not found."
        app_logger.error(err_msg)
        return jsonify({'message': err_msg}), 404
    except ValueError as e:
        err_msg = f"Data or Config Error: {e}"
        app_logger.error(err_msg, exc_info=False)
        return jsonify({'message': err_msg}), 400
    except RuntimeError as e:
        err_msg = f"Model Training Error: {e}"
        app_logger.error(err_msg, exc_info=True)
        return jsonify({'message': err_msg}), 500
    except IOError as e:
        err_msg = f"Model Saving Failed: {e}"
        app_logger.error(err_msg, exc_info=True)
        return jsonify({'message': "Failed to save model. Check server permissions/disk space."}), 500
    except Exception as e:
        err_msg = "Unexpected server error during training."
        app_logger.error(f"{err_msg} Details: {e}", exc_info=True)
        return jsonify({'message': "Unexpected server error. Check logs."}), 500

# --- Plotly JSON Generation Functions ---

def generate_label_distribution_plotly_json(labels, dataset_name):
    """Creates Plotly JSON for a horizontal bar chart of label frequencies."""
    if labels is None or len(labels) == 0:
        app_logger.warning(f"No labels for distribution plot: '{dataset_name}'.")
        return None
    try:
        label_series = pd.Series(labels)
        counts = label_series.value_counts().sort_index()
        fig = go.Figure(go.Bar(
            y=counts.index.astype(str), # Ensure y-axis labels are strings
            x=counts.values,
            orientation='h',
            marker_color='rgba(31, 119, 180, 0.8)',
            hoverinfo='x+y',
            hovertemplate='Sign: %{y}<br>Count: %{x}<extra></extra>',
        ))
        fig.update_layout(
            title=f'Label Distribution: {os.path.basename(dataset_name)}',
            xaxis_title='Number of Samples (Frequency)',
            yaxis_title='ASL Sign (Label)',
            yaxis={'categoryorder': 'total descending', 'type': 'category'}, # Ensure category axis
            height=max(400, len(counts) * 25),
            margin=dict(l=100, r=30, t=60, b=40),
            template="plotly_white",
            bargap=0.2
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        app_logger.error(f"Failed to generate Plotly distribution chart: {e}", exc_info=True)
        return None

def generate_data_split_plotly_json(train_count, test_count, dataset_name):
    """Creates Plotly JSON for a bar chart of Train vs Test split counts."""
    if train_count is None or test_count is None or (train_count == 0 and test_count == 0):
        app_logger.warning(f"Invalid counts for data split plot: {dataset_name} (Train: {train_count}, Test: {test_count})")
        return None
    try:
        labels_cat = ['Training Set', 'Test Set']
        counts = [train_count, test_count]
        total = sum(counts)
        percentages = [(c / total * 100) if total > 0 else 0 for c in counts]

        fig = go.Figure(go.Bar(
            x=labels_cat,
            y=counts,
            marker_color=['rgba(44, 160, 44, 0.8)', 'rgba(255, 127, 14, 0.8)'],
            text=[f'{p:.1f}% ({c})' for p, c in zip(percentages, counts)],
            textposition='auto',
            hoverinfo='y+text',
            hovertemplate='%{label}<br>Count: %{y}<extra></extra>',
        ))
        fig.update_layout(
            title=f'Data Split (Train/Test): {os.path.basename(dataset_name)}',
            xaxis_title='Dataset Split',
            yaxis_title='Number of Samples',
            template="plotly_white",
            margin=dict(l=50, r=20, t=60, b=40),
            uniformtext_minsize=8, uniformtext_mode='hide'
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        app_logger.error(f"Failed to generate Plotly data split chart: {e}", exc_info=True)
        return None

def generate_accuracy_comparison_plotly_json(results_list, dataset_name):
    """Creates Plotly JSON comparing model accuracies."""
    if not results_list:
        app_logger.warning(f"No results for accuracy comparison chart: '{dataset_name}'.")
        return None
    try:
        # Prepare data, handling None accuracy
        valid_results = [r for r in results_list if r.get('accuracy') is not None]
        if not valid_results:
             app_logger.warning(f"No valid accuracy values for comparison chart: '{dataset_name}'.")
             return None

        valid_results.sort(key=lambda x: x.get('accuracy', -1), reverse=True) # Sort best first

        model_names = [r.get('model_name', r.get('algorithm_key', 'Unknown')) for r in valid_results]
        accuracies = [r['accuracy'] * 100 for r in valid_results] # Convert to percentage

        colors = ['rgba(44, 160, 44, 0.8)' if acc >= 80 else # Green
                  'rgba(255, 193, 7, 0.8)' if acc >= 60 else # Yellow
                  'rgba(220, 53, 69, 0.8)' for acc in accuracies] # Red

        fig = go.Figure(go.Bar(
            x=model_names,
            y=accuracies,
            marker_color=colors,
            text=[f'{acc:.2f}%' for acc in accuracies],
            textposition='auto',
            hoverinfo='y',
            hovertemplate='Model: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>',
        ))
        fig.update_layout(
            title=f'Model Accuracy Comparison: {os.path.basename(dataset_name)}',
            xaxis_title='Model Algorithm',
            yaxis_title='Test Accuracy (%)',
            yaxis_range=[0, 105], # Ensure y-axis goes slightly above 100
            template="plotly_white",
            margin=dict(l=50, r=20, t=60, b=min(150, max(80, len(model_names)*15 ))), # Dynamic bottom margin
            xaxis={'categoryorder': 'total descending'}, # Order bars by accuracy
            uniformtext_minsize=8, uniformtext_mode='hide'
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        app_logger.error(f"Failed to generate Plotly accuracy comparison chart: {e}", exc_info=True)
        return None

def generate_time_comparison_plotly_json(results_list, dataset_name):
    """Creates Plotly JSON comparing model training times."""
    if not results_list:
        app_logger.warning(f"No results for time comparison chart: '{dataset_name}'.")
        return None
    try:
        # Prepare data, handling potential None times
        valid_results = [r for r in results_list if r.get('training_time') is not None]
        if not valid_results:
             app_logger.warning(f"No valid training time values for comparison chart: '{dataset_name}'.")
             return None

        valid_results.sort(key=lambda x: x.get('training_time', float('inf'))) # Sort fastest first

        model_names = [r.get('model_name', r.get('algorithm_key', 'Unknown')) for r in valid_results]
        times = [r['training_time'] for r in valid_results]

        fig = go.Figure(go.Bar(
            x=model_names,
            y=times,
            marker_color='rgba(23, 162, 184, 0.8)', # Info blue color
            text=[f'{t:.3f}s' for t in times],
            textposition='auto',
            hoverinfo='y',
            hovertemplate='Model: %{x}<br>Time: %{y:.3f} sec<extra></extra>',
        ))
        fig.update_layout(
            title=f'Model Training Time Comparison: {os.path.basename(dataset_name)}',
            xaxis_title='Model Algorithm',
            yaxis_title='Training Time (seconds)',
            template="plotly_white",
            margin=dict(l=50, r=20, t=60, b=min(150, max(80, len(model_names)*15 ))), # Dynamic bottom margin
            xaxis={'categoryorder': 'total ascending'}, # Order bars by time
            uniformtext_minsize=8, uniformtext_mode='hide'
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        app_logger.error(f"Failed to generate Plotly time comparison chart: {e}", exc_info=True)
        return None

# --- Comparison Route Logic ---
@app.route('/compare_models_run', methods=['POST'])
def handle_compare_models():
    """Handles AJAX request to compare models and returns results + chart JSON."""
    app_logger.info("Received POST request to /compare_models_run")
    if not request.is_json:
        app_logger.warning("Comparison rejected: Not JSON.")
        return jsonify({'message': 'Request must be application/json.'}), 415

    data = request.get_json()
    dataset_filename = data.get('dataset')
    algorithm_keys = data.get('algorithms')

    if not dataset_filename: return jsonify({'message': 'Missing parameter: dataset.'}), 400
    if not algorithm_keys or not isinstance(algorithm_keys, list) or not algorithm_keys:
         return jsonify({'message': 'Missing/empty parameter: algorithms (list required).'}), 400

    dataset_path = os.path.join(DATASET_DIR, dataset_filename)
    app_logger.info(f"Starting comparison: Dataset='{dataset_filename}', Algos='{', '.join(algorithm_keys)}'")

    results_list = []
    warnings_list = []
    first_successful_split = {'train_count': None, 'test_count': None} # To generate split chart

    try:
        data_features, labels = load_data(dataset_path)
        app_logger.info(f"Loaded '{dataset_filename}': {data_features.shape[0]} samples, {data_features.shape[1]} features.")

        # Feature dimension check (just warning for comparison)
        if data_features.shape[1] != EXPECTED_LANDMARK_FEATURES:
             warn_msg = (f"Warning: Dataset '{dataset_filename}' has {data_features.shape[1]} features, "
                         f"config expects {EXPECTED_LANDMARK_FEATURES}. Results might be affected.")
             app_logger.warning(warn_msg)
             warnings_list.append(warn_msg)

        for algo_key in algorithm_keys:
            if algo_key not in ALGORITHM_IMPLEMENTATIONS:
                msg = f"Algorithm key '{algo_key}' not recognized/skipped."
                app_logger.warning(msg)
                warnings_list.append(msg)
                continue

            model_class, model_params = ALGORITHM_IMPLEMENTATIONS[algo_key]
            # Map key back to display name (or use class name)
            model_display_names = { # Duplicated from /compare route - refactor possibility
                'random forest model': 'Random Forest', 'logistic regression model': 'Logistic Regression',
                'svm model': 'SVM', 'naive bayes model': 'Naive Bayes',
                'decision tree model': 'Decision Tree', 'lda model': 'LDA'
            }
            model_display_name = model_display_names.get(algo_key, model_class.__name__)
            app_logger.info(f"Comparing: {model_display_name} ({algo_key})")

            try:
                model_instance = model_class(**model_params)
                # Use the core train_model function (does NOT save model here)
                _, accuracy, duration, train_count, test_count = train_model(
                    data_features, labels, model_instance
                )

                results_list.append({
                    'algorithm_key': algo_key,
                    'model_name': model_display_name, # Use friendly name for table/charts
                    'accuracy': accuracy, # Store None if full dataset training happened
                    'training_time': duration,
                    'train_count': train_count,
                    'test_count': test_count
                })

                # Store split info from first successful run for the chart
                if accuracy is not None and first_successful_split['test_count'] is None and test_count > 0 :
                    first_successful_split['train_count'] = train_count
                    first_successful_split['test_count'] = test_count

                acc_disp = f"{accuracy*100:.2f}%" if accuracy is not None else "N/A (Full dataset)"
                app_logger.info(f"Finished {model_display_name}. Accuracy: {acc_disp}, Time: {duration:.3f}s")

            except (ValueError, RuntimeError, TypeError) as train_err:
                err_msg = f"Failed to train/evaluate '{model_display_name}': {train_err}"
                app_logger.error(err_msg, exc_info=False)
                warnings_list.append(err_msg)
                # Optionally add a 'failed' entry to results_list for table consistency?
                results_list.append({'algorithm_key': algo_key, 'model_name': model_display_name, 'accuracy': None, 'training_time': None, 'train_count': None, 'test_count': None})


        # Generate comparison and context charts *after* all models run
        accuracy_comp_json = generate_accuracy_comparison_plotly_json(results_list, dataset_filename)
        time_comp_json = generate_time_comparison_plotly_json(results_list, dataset_filename)
        dist_plot_json = generate_label_distribution_plotly_json(labels, dataset_filename)

        # Generate split chart only if a successful split occurred
        if first_successful_split['test_count'] is not None and first_successful_split['test_count'] > 0:
            split_plot_json = generate_data_split_plotly_json(
                first_successful_split['train_count'],
                first_successful_split['test_count'],
                dataset_filename)
        else:
            split_plot_json = None
            warnings_list.append("Data split chart could not be generated (likely due to small class sizes preventing a test split).")

        app_logger.info(f"Comparison finished for '{dataset_filename}'. {len(results_list)} results, {len(warnings_list)} warnings.")

        return jsonify({
            'results': results_list,
            'warnings': warnings_list,
            'accuracy_comparison_plot_json': accuracy_comp_json,
            'time_comparison_plot_json': time_comp_json,
            'distribution_plot_json': dist_plot_json,
            'data_split_plot_json': split_plot_json
        }), 200

    except FileNotFoundError:
        err_msg = f"Dataset file '{dataset_filename}' not found."
        app_logger.error(err_msg)
        return jsonify({'message': err_msg}), 404
    except (ValueError, RuntimeError, TypeError) as load_err:
        err_msg = f"Error processing dataset '{dataset_filename}': {load_err}"
        app_logger.error(err_msg, exc_info=True)
        return jsonify({'message': err_msg}), 400
    except Exception as e:
        err_msg = "Unexpected server error during comparison."
        app_logger.error(f"{err_msg} Details: {e}", exc_info=True)
        return jsonify({'message': err_msg}), 500

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handles new client connections, attempts to load default model."""
    sid = request.sid
    app_logger.info(f"Client connected: {sid}")
    session_data = get_session_data(sid)
    current_model_name = session_data.get('model_name')
    status_msg = "Connected."
    loaded_ok = False
    model_info_for_emit = None

    if not current_model_name:
        app_logger.info(f"SID {sid}: No model loaded. Attempting auto-load.")
        available_models = get_available_items(MODEL_DIR, '.pkl')
        model_to_load = None
        if DEFAULT_MODEL_NAME in available_models: model_to_load = DEFAULT_MODEL_NAME
        elif available_models: model_to_load = available_models[0]

        if model_to_load:
            loaded_ok, load_msg = load_model(sid, model_to_load)
            model_info_for_emit = session_data.get('model_name')
            if loaded_ok:
                status_msg = f"Connected. Auto-loaded: '{model_info_for_emit}'"
                if "Warning:" in load_msg: status_msg += f" ({load_msg.split('Warning: ')[1]})"
            else:
                status_msg = f"Connected. Auto-load failed for '{model_to_load}'. {load_msg}"
        else:
            status_msg = "Connected. No models found. Please train a model."
            model_info_for_emit = None
            app_logger.warning(f"SID {sid}: No models available in {MODEL_DIR}.")
    else:
        status_msg = f"Reconnected. Using model: '{current_model_name}'"
        model_info_for_emit = current_model_name
        loaded_ok = True

    # Emit initial state
    emit('status_update', {'message': status_msg, 'current_model': model_info_for_emit})
    emit('prediction_result', {'prediction': 'None', 'sequence': 'None', 'current_model': model_info_for_emit})


@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections and cleans up session data."""
    sid = request.sid
    if sid in user_sessions:
        start_time = user_sessions[sid].get('connect_time', time.time())
        duration = time.time() - start_time
        app_logger.info(f"Client disconnected: {sid}. Duration: {duration:.2f}s. Cleaning up.")
        del user_sessions[sid]
    else:
        app_logger.info(f"Client disconnected: {sid}. No session found.")


@socketio.on('select_model')
def handle_select_model(data):
    """Handles user request to switch the prediction model."""
    sid = request.sid
    session_data = get_session_data(sid)
    new_model_name = data.get('model_name')
    current_model_name = session_data.get('model_name')

    app_logger.info(f"SID {sid}: Request select model '{new_model_name}'.")
    if not new_model_name:
        emit('status_update', {'message': "Invalid model select (no name).", 'current_model': current_model_name})
        return

    available_models = get_available_items(MODEL_DIR, '.pkl')
    if new_model_name not in available_models:
        emit('status_update', {'message': f"Model '{new_model_name}' not found.", 'current_model': current_model_name})
        return

    loaded_ok, load_msg = load_model(sid, new_model_name)
    updated_model_name = session_data.get('model_name') # Refresh after load attempt

    if loaded_ok:
        app_logger.info(f"SID {sid}: Switched to model '{updated_model_name}'. Resetting state.")
        session_data['sequence'] = []
        session_data['last_prediction'] = None
        session_data['last_valid_prediction_time'] = time.time()
        emit('status_update', {'message': load_msg, 'current_model': updated_model_name})
        emit('prediction_result', {'prediction': 'None', 'sequence': 'None', 'current_model': updated_model_name})
    else:
        # Revert if needed - load_model clears on fail
        session_data['model_name'] = current_model_name
        # You might need to re-fetch the model object if load_model cleared it, potentially from a global cache or reload it
        app_logger.error(f"SID {sid}: Failed to load '{new_model_name}': {load_msg}. Keeping: '{current_model_name}'.")
        emit('status_update', {'message': f"Error loading '{new_model_name}': {load_msg}. Keeping: {current_model_name or 'None'}.", 'current_model': current_model_name})


@socketio.on('process_frame')
def handle_process_frame(data):
    """Processes a single frame from the real-time video feed."""
    sid = request.sid
    session_data = get_session_data(sid)
    current_model_name = session_data.get('model_name')
    log_prefix = "RTP:"

    if mediapipe_init_error:
        if time.time() - session_data.get('last_mediapipe_error_log_time', 0) > 60:
             app_logger.warning(f"{log_prefix} SID {sid}: Skipping frame, MediaPipe inactive ({mediapipe_init_error}).")
             session_data['last_mediapipe_error_log_time'] = time.time()
             emit('status_update', {'message': 'Error: Processing disabled (MediaPipe init failed).', 'current_model': current_model_name}) # Inform client
        return

    image_b64 = data.get('image')
    if not image_b64: return # Ignore empty

    current_sequence_str = ''.join(session_data.get('sequence', [])) or 'None'
    if not current_model_name:
         if session_data.get('last_emitted_status') != 'No Model':
             emit('prediction_result', {'prediction': 'No Model', 'sequence': current_sequence_str, 'current_model': None})
             session_data['last_emitted_status'] = 'No Model'
         return

    landmarks_array = _process_image_get_landmarks(sid, image_b64, log_prefix=log_prefix)
    predicted_char = "No Hand" if landmarks_array is None else predict_from_landmarks(sid, landmarks_array)
    if predicted_char in ["Input Error", "Predict Error"]: predicted_char = "Error"

    now = time.time()
    last_prediction = session_data.get('last_prediction')
    sequence = session_data.get('sequence', [])
    is_valid_sign = predicted_char not in ["None", "No Hand", "Error", "No Model"]

    if is_valid_sign:
        if predicted_char != last_prediction:
            sequence.append(predicted_char)
            session_data['last_prediction'] = predicted_char
            app_logger.info(f"{log_prefix} SID {sid}: Appended '{predicted_char}'. Seq: {''.join(sequence)}")
        session_data['last_valid_prediction_time'] = now
    else:
        last_valid_time = session_data.get('last_valid_prediction_time', now)
        if (now - last_valid_time > NO_HAND_TIMEOUT) and sequence:
            app_logger.info(f"{log_prefix} SID {sid}: Timeout ({NO_HAND_TIMEOUT}s). Clearing seq: {''.join(sequence)}")
            sequence = []
            session_data['last_prediction'] = None
            session_data['last_valid_prediction_time'] = now
        # Only update timestamp if no valid sign, prevent timer reset
        elif not sequence : # If sequence is empty, update timestamp regardless to avoid rapid clearing checks
             session_data['last_valid_prediction_time'] = now


    session_data['sequence'] = sequence
    updated_sequence_str = ''.join(sequence) if sequence else 'None'
    session_data['last_emitted_status'] = predicted_char

    emit('prediction_result', {
        'prediction': predicted_char,
        'sequence': updated_sequence_str,
        'current_model': current_model_name
    })


@socketio.on('clear_sequence')
def handle_clear_sequence():
    """Handles client request to manually clear the recognized sequence."""
    sid = request.sid
    session_data = get_session_data(sid)
    app_logger.info(f"SID {sid}: Clearing sequence by request.")
    session_data['sequence'] = []
    session_data['last_prediction'] = None
    session_data['last_valid_prediction_time'] = time.time()
    emit('prediction_result', {
        'prediction': 'None', 'sequence': 'None',
        'current_model': session_data.get('model_name')
    })
    emit('status_update', {'message': 'Sequence cleared.', 'current_model': session_data.get('model_name')})


@socketio.on('process_image')
def handle_process_image(data):
    """Processes a single uploaded static image."""
    sid = request.sid
    session_data = get_session_data(sid)
    current_model_name = session_data.get('model_name')
    log_prefix = "IMG:"

    if mediapipe_init_error:
        app_logger.warning(f"{log_prefix} SID {sid}: Skipping image, MediaPipe inactive ({mediapipe_init_error}).")
        emit('prediction_result', {
            'imageIndex': data.get('imageIndex', -1),
            'originalIndex': data.get('originalIndex', -1),
            'prediction': 'Processing Error', # Generic processing error
            'current_model': current_model_name
        })
        return

    image_unique_id = data.get('imageIndex')
    original_file_index = data.get('originalIndex')
    image_b64 = data.get('image')

    # Combine checks using 'or' for cleaner validation
    if image_unique_id is None or original_file_index is None or not isinstance(image_b64, str) or not image_b64:
        app_logger.warning(f"{log_prefix} SID {sid}: Invalid process_image request (ID: {image_unique_id}, OrigIdx: {original_file_index}, ImgEmpty: {not image_b64})")
        emit('prediction_result', {
            'imageIndex': image_unique_id if image_unique_id is not None else -1,
            'originalIndex': original_file_index if original_file_index is not None else -1,
            'prediction': 'Bad Request',
            'current_model': current_model_name
        })
        return

    if not current_model_name:
        emit('prediction_result', {
            'imageIndex': image_unique_id, 'originalIndex': original_file_index,
            'prediction': 'No Model', 'current_model': None
        })
        return

    app_logger.info(f"{log_prefix} SID {sid}: Processing image {original_file_index} (ID: {image_unique_id}) with '{current_model_name}'...")
    landmarks_array = _process_image_get_landmarks(sid, image_b64, log_prefix=log_prefix)

    predicted_char = "No Hand"
    if landmarks_array is not None:
        predicted_char = predict_from_landmarks(sid, landmarks_array)
    elif landmarks_array is None and mediapipe_init_error: # Catch case where landmarks failed specifically due to mediapipe issue
         predicted_char = "Processing Error"

    app_logger.info(f"{log_prefix} SID {sid}: Image {original_file_index} (ID: {image_unique_id}) -> Result: '{predicted_char}'")

    emit('prediction_result', {
        'imageIndex': image_unique_id,
        'originalIndex': original_file_index,
        'prediction': predicted_char,
        'current_model': current_model_name
    })


# --- Main Execution Block ---
if __name__ == '__main__':
    print("-" * 60)
    print(f"Flask App Path: {app.root_path}")
    print(f"Static Folder: {app.static_folder}")
    print(f"Template Folder: {app.template_folder}") # Should be the project root based on Flask init
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Async Mode: {async_mode}")
    print("-" * 60)
    # use_reloader=False is important for production and some debug scenarios to avoid double initialization
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    # Added log_output=True to potentially see more server logs directly in console

# --- END OF FILE main.py ---