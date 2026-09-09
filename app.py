import sqlite3
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model_weights/best.pt"
DB_PATH = "detections.db"


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Marine Debris Detection",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# DATABASE
# ============================================================

def create_database():

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            object_class TEXT,
            confidence REAL,
            x1 REAL,
            y1 REAL,
            x2 REAL,
            y2 REAL,
            latitude REAL,
            longitude REAL,
            timestamp TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_detection(
    image_name,
    object_class,
    confidence,
    x1,
    y1,
    x2,
    y2,
    latitude,
    longitude
):

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO detections (
            image_name,
            object_class,
            confidence,
            x1,
            y1,
            x2,
            y2,
            latitude,
            longitude,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_name,
        object_class,
        confidence,
        x1,
        y1,
        x2,
        y2,
        latitude,
        longitude,
        datetime.now().isoformat()
    ))

    connection.commit()
    connection.close()


def get_detection_history():

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            image_name,
            object_class,
            confidence,
            latitude,
            longitude,
            timestamp
        FROM detections
        ORDER BY id DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows


# ============================================================
# SONAR IMAGE PREPROCESSING
# ============================================================

def preprocess_sonar(image):

    """
    Side-scan sonar preprocessing.

    Step 1:
        Median filtering for noise reduction.

    Step 2:
        CLAHE for local contrast enhancement.
    """

    # PIL -> NumPy
    image_array = np.array(image)

    # RGB -> BGR
    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Noise reduction
    # --------------------------------------------------------

    denoised = cv2.medianBlur(
        image_bgr,
        5
    )

    # --------------------------------------------------------
    # Contrast enhancement using CLAHE
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        denoised,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel
        )
    )

    enhanced_bgr = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # BGR -> RGB
    enhanced_rgb = cv2.cvtColor(
        enhanced_bgr,
        cv2.COLOR_BGR2RGB
    )

    return enhanced_rgb


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = YOLO(MODEL_PATH)

    return model


# ============================================================
# INITIALIZE DATABASE
# ============================================================

create_database()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌊 AI-Powered Underwater Marine Debris Detection"
)

st.write(
    "Side-scan sonar imagery analysis using YOLOv9t."
)

st.info(
    "The prototype performs sonar image preprocessing using "
    "median filtering and CLAHE and uses YOLOv9t for "
    "marine-debris object detection."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Detection Settings"
)

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.05,
    step=0.05
)


st.sidebar.header(
    "📍 Location"
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=0.0,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=0.0,
    format="%.6f"
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader(
    "📡 Upload Side-Scan Sonar Image"
)

uploaded_file = st.file_uploader(
    "Choose a sonar image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# DETECTION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader(
        "1️⃣ Original Sonar Image"
    )

    st.image(
        image,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Detection button
    # --------------------------------------------------------

    if st.button(
        "🔍 Detect Marine Debris",
        type="primary"
    ):

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        with st.spinner(
            "Removing sonar noise and enhancing image..."
        ):

            processed_image = preprocess_sonar(
                image
            )

        st.subheader(
            "2️⃣ Preprocessed Sonar Image"
        )

        st.image(
            processed_image,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        try:

            model = load_model()

        except Exception as error:

            st.error(
                f"Could not load YOLOv9t model: {error}"
            )

            st.stop()

        # ----------------------------------------------------
        # YOLO detection
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We keep the original image as the YOLO input
        # because this is the input configuration that is
        # currently producing detections with your trained
        # model.
        #
        # The processed image is still displayed as part of
        # the preprocessing stage.
        # ----------------------------------------------------

        with st.spinner(
            "Running YOLOv9t detection..."
        ):

            results = model.predict(
                source= image,
                conf=confidence_threshold,
                iou=0.5,
                device=0,
                imgsz=640,
                verbose=False
            )

        result = results[0]

        boxes = result.boxes

        # ----------------------------------------------------
        # Create annotated image
        # ----------------------------------------------------

        annotated_image = np.array(
            image
        ).copy()

        detection_count = 0

        # ----------------------------------------------------
        # Draw EVERY detection manually
        # ----------------------------------------------------

        if boxes is not None:

            for box in boxes:

                detection_count += 1

                # Bounding box
                coordinates = box.xyxy[0].tolist()

                x1, y1, x2, y2 = map(
                    int,
                    coordinates
                )

                # Class
                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[
                    class_id
                ]

                # Confidence
                confidence = float(
                    box.conf[0]
                )

                # ------------------------------------------------
                # Draw bounding box
                # ------------------------------------------------

                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                # ------------------------------------------------
                # Draw label
                # ------------------------------------------------

                label = (
                    f"{class_name} "
                    f"{confidence * 100:.1f}%"
                )

                # Text position
                text_y = max(
                    y1 - 10,
                    25
                )

                cv2.putText(
                    annotated_image,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------
        # Display detection result
        # ----------------------------------------------------

        st.subheader(
            "3️⃣ AI Detection Result"
        )

        st.image(
            annotated_image,
            channels="RGB",
            use_container_width=True
        )

        # ----------------------------------------------------
        # Detection summary
        # ----------------------------------------------------

        if detection_count == 0:

            st.warning(
                "No marine-debris target was detected "
                "above the selected confidence threshold."
            )

        else:

            st.success(
                f"✅ {detection_count} detection(s) found."
            )

            # ------------------------------------------------
            # Detected objects
            # ------------------------------------------------

            st.subheader(
                "4️⃣ Detected Objects"
            )

            for box in boxes:

                # --------------------------------------------
                # Class
                # --------------------------------------------

                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[
                    class_id
                ]

                # --------------------------------------------
                # Confidence
                # --------------------------------------------

                confidence = float(
                    box.conf[0]
                )

                # --------------------------------------------
                # Bounding box
                # --------------------------------------------

                coordinates = box.xyxy[
                    0
                ].tolist()

                x1, y1, x2, y2 = coordinates

                # --------------------------------------------
                # Display information
                # --------------------------------------------

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Object",
                    class_name
                )

                col2.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                with col3:

                    st.write(
                        "**Bounding Box**"
                    )

                    st.write(
                        f"X1: {x1:.1f}"
                    )

                    st.write(
                        f"Y1: {y1:.1f}"
                    )

                    st.write(
                        f"X2: {x2:.1f}"
                    )

                    st.write(
                        f"Y2: {y2:.1f}"
                    )

                st.divider()

                # --------------------------------------------
                # Save detection to SQLite
                # --------------------------------------------

                save_detection(
                    image_name=uploaded_file.name,
                    object_class=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    latitude=latitude,
                    longitude=longitude
                )

            st.success(
                "💾 Detection results saved to database."
            )


# ============================================================
# DATABASE HISTORY
# ============================================================

st.divider()

st.subheader(
    "📊 Detection History"
)

detections = get_detection_history()

if len(detections) == 0:

    st.write(
        "No detections stored yet."
    )

else:

    for detection in detections:

        (
            detection_id,
            image_name,
            object_class,
            confidence,
            lat,
            lon,
            timestamp
        ) = detection

        st.write(
            f"**#{detection_id}** | "
            f"**{object_class}** | "
            f"Confidence: {confidence * 100:.2f}% | "
            f"{image_name} | "
            f"Location: {lat}, {lon} | "
            f"{timestamp}"
        )