import os
import io
import zipfile

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CROWDGUARD CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best (1).pt"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CrowdGuard",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error("❌ CrowdGuard model not found!")
        st.error(f"Expected location: {MODEL_PATH}")
        st.stop()

    model = YOLO(MODEL_PATH)

    return model


model = load_model()


# ============================================================
# CROWD ANALYSIS FUNCTION
# ============================================================

def analyze_image(image):

    # Make sure image is RGB
    image = image.convert("RGB")

    # --------------------------------------------------------
    # IMAGE DIMENSIONS
    # --------------------------------------------------------

    width, height = image.size

    area = width * height

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # COUNT PEOPLE
    # --------------------------------------------------------

    people_count = 0

    if result.boxes is not None:

        for cls in result.boxes.cls:

            class_id = int(cls)

            # Your model:
            # 0 = person

            if class_id == 0:
                people_count += 1

    # --------------------------------------------------------
    # DENSITY
    # --------------------------------------------------------

    density = (people_count / area) * 100000

    # --------------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------------

    if density < 1.0:

        risk_level = "LOW"

        alert = "🟢 LOW CROWD - SAFE"

    elif density < 2.5:

        risk_level = "MODERATE"

        alert = "⚠️ MODERATE CROWD"

    else:

        risk_level = "HIGH"

        alert = "🚨 HIGH CROWD - IMMEDIATE ATTENTION"

    # --------------------------------------------------------
    # DRAW DETECTION BOXES
    # --------------------------------------------------------

    annotated_array = result.plot()

    annotated_image = Image.fromarray(annotated_array)

    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return {
        "image": annotated_image,
        "width": width,
        "height": height,
        "area": area,
        "people": people_count,
        "density": density,
        "risk": risk_level,
        "alert": alert
    }


# ============================================================
# DISPLAY RESULTS FUNCTION
# IMPORTANT:
# THIS MUST BE DEFINED BEFORE IT IS CALLED
# ============================================================

def display_results(results):

    st.divider()

    st.header("🛡️ CrowdGuard Analysis")

    # --------------------------------------------------------
    # DETECTION IMAGE
    # --------------------------------------------------------

    st.subheader("🎯 Detection Result")

    st.image(
        results["image"],
        caption="CrowdGuard Person Detection",
        width=800
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader("📊 Crowd Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 People Detected",
            results["people"]
        )

    with col2:

        st.metric(
            "📐 Image Area",
            f'{results["area"]:,} px²'
        )

    with col3:

        st.metric(
            "📊 Density",
            f'{results["density"]:.3f}'
        )

    with col4:

        st.metric(
            "⚠️ Risk Level",
            results["risk"]
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    st.divider()

    st.subheader("🚨 CROWDGUARD RESULT")

    st.write(
        f'👥 **People Detected:** {results["people"]}'
    )

    st.write(
        f'📐 **Image Area:** {results["area"]:,} pixels²'
    )

    st.write(
        f'📊 **Density Score:** '
        f'{results["density"]:.3f} people / 100,000 pixels'
    )

    st.write(
        f'⚠️ **Risk Level:** {results["risk"]}'
    )

    # --------------------------------------------------------
    # RISK MESSAGE
    # --------------------------------------------------------

    if results["risk"] == "HIGH":

        st.error(results["alert"])

    elif results["risk"] == "MODERATE":

        st.warning(results["alert"])

    else:

        st.success(results["alert"])


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ CrowdGuard")

st.write(
    "AI-powered crowd detection and crowd-risk analysis "
    "using a trained YOLO model."
)

st.caption(
    "Model: CrowdHuman-trained YOLO | Class: person"
)

st.divider()


# ============================================================
# MODEL STATUS
# ============================================================

with st.expander("🧠 Model Information"):

    st.write("✅ CrowdGuard model loaded")

    st.write(f"📍 **Model:** `{MODEL_PATH}`")

    st.write(f"🧠 **Classes:** `{model.names}`")


# ============================================================
# IMAGE SOURCE
# ============================================================

st.subheader("📷 Choose Image Source")

source = st.radio(
    "How do you want to provide the image?",
    [
        "📦 Select Image From ZIP",
        "📤 Upload Individual Image",
        "📁 Select Local Image"
    ],
    horizontal=True
)


# ============================================================
# OPTION 1
# SELECT ONE IMAGE DIRECTLY FROM A LOCAL ZIP
# ============================================================

if source == "📦 Select Image From ZIP":

    st.info(
        "📦 Your CrowdHuman ZIP remains on your computer. "
        "CrowdGuard reads the ZIP and extracts ONLY the image "
        "you select."
    )

    st.write("### 1️⃣ Enter the CrowdHuman ZIP path")

    zip_path = st.text_input(
        "ZIP file path",
        placeholder=(
            r"C:\Users\HP\Downloads\crowdhuman.v1i.yolov11.zip"
        )
    )

    if zip_path:

        # Remove accidental quotation marks
        zip_path = zip_path.strip().strip('"')

        # ----------------------------------------------------
        # CHECK ZIP
        # ----------------------------------------------------

        if not os.path.exists(zip_path):

            st.error(
                "❌ ZIP file not found."
            )

            st.info(
                "Check the Windows path carefully."
            )

        elif not zipfile.is_zipfile(zip_path):

            st.error(
                "❌ The selected file is not a valid ZIP file."
            )

        else:

            st.success(
                "✅ CrowdHuman ZIP found!"
            )

            # ------------------------------------------------
            # READ IMAGE NAMES FROM ZIP
            # ------------------------------------------------

            try:

                with zipfile.ZipFile(
                    zip_path,
                    "r"
                ) as zip_file:

                    image_files = []

                    for name in zip_file.namelist():

                        if name.lower().endswith(
                            (
                                ".jpg",
                                ".jpeg",
                                ".png",
                                ".webp"
                            )
                        ):

                            image_files.append(name)

                # ------------------------------------------------
                # SHOW IMAGE COUNT
                # ------------------------------------------------

                st.write(
                    f"🖼️ **Images found in ZIP:** "
                    f"{len(image_files):,}"
                )

                if len(image_files) == 0:

                    st.error(
                        "❌ No image files were found inside the ZIP."
                    )

                else:

                    # ------------------------------------------------
                    # SEARCH IMAGE
                    # ------------------------------------------------

                    st.write("### 2️⃣ Find your image")

                    search_text = st.text_input(
                        "🔎 Search by image filename",
                        placeholder="Example: 273271"
                    )

                    # ------------------------------------------------
                    # FILTER
                    # ------------------------------------------------

                    if search_text:

                        matching_images = [
                            image_name
                            for image_name in image_files
                            if search_text.lower()
                            in image_name.lower()
                        ]

                    else:

                        matching_images = image_files

                    st.write(
                        f"🔍 Matching images: "
                        f"**{len(matching_images):,}**"
                    )

                    if len(matching_images) == 0:

                        st.warning(
                            "⚠️ No matching image found."
                        )

                    else:

                        # ------------------------------------------------
                        # SELECT ONE IMAGE
                        # ------------------------------------------------

                        st.write("### 3️⃣ Select ONE image")

                        selected_image = st.selectbox(
                            "Image",
                            matching_images
                        )

                        st.caption(
                            f"📍 `{selected_image}`"
                        )

                        # ------------------------------------------------
                        # READ ONLY SELECTED IMAGE
                        # ------------------------------------------------

                        with zipfile.ZipFile(
                            zip_path,
                            "r"
                        ) as zip_file:

                            image_bytes = zip_file.read(
                                selected_image
                            )

                        image = Image.open(
                            io.BytesIO(image_bytes)
                        ).convert("RGB")

                        # ------------------------------------------------
                        # SHOW SELECTED IMAGE
                        # ------------------------------------------------

                        st.write("### 4️⃣ Selected Image")

                        st.image(
                            image,
                            caption=selected_image,
                            width=700
                        )

                        # ------------------------------------------------
                        # ANALYZE
                        # ------------------------------------------------

                        if st.button(
                            "🔍 Analyze Selected Image",
                            type="primary",
                            use_container_width=True
                        ):

                            with st.spinner(
                                "🧠 CrowdGuard is detecting people..."
                            ):

                                analysis_results = analyze_image(
                                    image
                                )

                            display_results(
                                analysis_results
                            )


            except Exception as error:

                st.error(
                    f"❌ Error reading ZIP: {error}"
                )


# ============================================================
# OPTION 2
# UPLOAD ONE INDIVIDUAL IMAGE
# ============================================================

elif source == "📤 Upload Individual Image":

    st.write(
        "Upload ONE image only."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption=uploaded_file.name,
            width=700
        )

        if st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "🧠 CrowdGuard is detecting people..."
            ):

                analysis_results = analyze_image(
                    image
                )

            display_results(
                analysis_results
            )


# ============================================================
# OPTION 3
# LOCAL IMAGE PATH
# ============================================================

elif source == "📁 Select Local Image":

    st.info(
        "Enter the complete Windows path of an image."
    )

    local_image_path = st.text_input(
        "Image path",
        placeholder=(
            r"C:\Users\HP\Desktop\CrowdGuard\data\videos\image.jpg"
        )
    )

    if local_image_path:

        local_image_path = (
            local_image_path
            .strip()
            .strip('"')
        )

        if not os.path.exists(local_image_path):

            st.error(
                "❌ Image not found."
            )

        else:

            try:

                image = Image.open(
                    local_image_path
                ).convert("RGB")

                st.image(
                    image,
                    caption=os.path.basename(
                        local_image_path
                    ),
                    width=700
                )

                if st.button(
                    "🔍 Analyze Local Image",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "🧠 CrowdGuard is detecting people..."
                    ):

                        analysis_results = analyze_image(
                            image
                        )

                    display_results(
                        analysis_results
                    )

            except Exception as error:

                st.error(
                    f"❌ Could not open image: {error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ CrowdGuard | YOLO-based Crowd Detection & Risk Analysis"
)