import io
import os
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CROWDGUARD CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best (1).pt"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CrowdGuard",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("❌ CrowdGuard model not found.")
        st.error(f"Expected location: {MODEL_PATH}")
        st.stop()

    try:
        model = YOLO(str(MODEL_PATH))
        return model
    except Exception as error:
        st.error(f"❌ Could not load CrowdGuard model: {error}")
        st.stop()


model = load_model()


# ============================================================
# CROWD ANALYSIS
# ============================================================

def analyze_image(image: Image.Image):
    """Run YOLO person detection and calculate crowd risk."""

    image = image.convert("RGB")

    width, height = image.size
    area = width * height

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False,
    )

    result = results[0]

    people_count = 0

    if result.boxes is not None and result.boxes.cls is not None:
        for cls in result.boxes.cls:
            class_id = int(cls)

            # CrowdGuard model:
            # 0 = person
            if class_id == 0:
                people_count += 1

    # Density score used by the current CrowdGuard system
    density = (people_count / area) * 100000

    if density < 1.0:
        risk_level = "LOW"
        alert = "🟢 LOW CROWD - SAFE"
    elif density < 2.5:
        risk_level = "MODERATE"
        alert = "⚠️ MODERATE CROWD"
    else:
        risk_level = "HIGH"
        alert = "🚨 HIGH CROWD - IMMEDIATE ATTENTION"

    annotated_array = result.plot()
    annotated_image = Image.fromarray(annotated_array)

    return {
        "image": annotated_image,
        "width": width,
        "height": height,
        "area": area,
        "people": people_count,
        "density": density,
        "risk": risk_level,
        "alert": alert,
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(results):
    st.divider()

    st.header("🛡️ CrowdGuard Analysis")

    st.subheader("🎯 Detection Result")

    st.image(
        results["image"],
        caption="CrowdGuard Person Detection",
        width=800,
    )

    st.subheader("📊 Crowd Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 People Detected",
            results["people"],
        )

    with col2:
        st.metric(
            "📐 Image Area",
            f'{results["area"]:,} px²',
        )

    with col3:
        st.metric(
            "📊 Density",
            f'{results["density"]:.3f}',
        )

    with col4:
        st.metric(
            "⚠️ Risk Level",
            results["risk"],
        )

    st.divider()

    st.subheader("🚨 CROWDGUARD RESULT")

    st.write(f'👥 **People Detected:** {results["people"]}')
    st.write(f'📐 **Image Area:** {results["area"]:,} pixels²')
    st.write(
        f'📊 **Density Score:** '
        f'{results["density"]:.3f} people / 100,000 pixels'
    )
    st.write(f'⚠️ **Risk Level:** {results["risk"]}')

    if results["risk"] == "HIGH":
        st.error(results["alert"])
    elif results["risk"] == "MODERATE":
        st.warning(results["alert"])
    else:
        st.success(results["alert"])


# ============================================================
# ANALYZE BUTTON HELPER
# ============================================================

def analyze_and_display(image):
    with st.spinner("🧠 CrowdGuard is detecting people..."):
        analysis_results = analyze_image(image)

    display_results(analysis_results)


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
    st.success("CrowdGuard model loaded")
    st.write(f"📍 **Model:** `{MODEL_PATH}`")
    st.write(f"🧠 **Classes:** `{model.names}`")


# ============================================================
# IMAGE SOURCE
# ============================================================

st.subheader("📷 Choose Image Source")

source = st.radio(
    "How do you want to provide the image?",
    [
        "📤 Upload Individual Image",
        "📦 Select Image From ZIP",
        "📁 Select Local Image",
    ],
    horizontal=True,
)


# ============================================================
# OPTION 1: UPLOAD ONE IMAGE
# ============================================================

if source == "📤 Upload Individual Image":

    st.info(
        "Upload one JPG, JPEG, PNG, or WEBP image "
        "and CrowdGuard will analyze it."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption=uploaded_file.name,
                width=700,
            )

            if st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True,
            ):
                analyze_and_display(image)

        except Exception as error:
            st.error(f"❌ Could not open image: {error}")


# ============================================================
# OPTION 2: SELECT ONE IMAGE FROM A LOCAL ZIP
# ============================================================

elif source == "📦 Select Image From ZIP":

    st.info(
        "💡 This ZIP-path option is for running CrowdGuard "
        "locally on your computer. Enter the path to your "
        "existing CrowdHuman ZIP. CrowdGuard reads only the "
        "image you select and does not extract the whole ZIP."
    )

    zip_path = st.text_input(
        "📦 CrowdHuman ZIP file path",
        placeholder=(
            r"C:\Users\HP\Downloads\crowdhuman.v1i.yolov11.zip"
        ),
    )

    if zip_path:

        zip_path = zip_path.strip().strip('"')

        if not os.path.isfile(zip_path):
            st.error("❌ ZIP file not found. Check the path.")

        elif not zipfile.is_zipfile(zip_path):
            st.error("❌ The selected file is not a valid ZIP.")

        else:
            st.success("✅ CrowdHuman ZIP found.")

            try:
                with zipfile.ZipFile(zip_path, "r") as zip_file:

                    image_files = [
                        name
                        for name in zip_file.namelist()
                        if name.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".webp")
                        )
                        and not name.endswith("/")
                    ]

                st.write(
                    f"🖼️ **Images found:** {len(image_files):,}"
                )

                if not image_files:
                    st.error("❌ No image files found inside the ZIP.")

                else:

                    search_text = st.text_input(
                        "🔎 Search image filename",
                        placeholder="Example: 273271",
                    )

                    if search_text:
                        matching_images = [
                            name
                            for name in image_files
                            if search_text.lower() in name.lower()
                        ]
                    else:
                        matching_images = image_files

                    st.write(
                        f"🔍 Matching images: "
                        f"**{len(matching_images):,}**"
                    )

                    if not matching_images:
                        st.warning("⚠️ No matching image found.")

                    else:

                        selected_image = st.selectbox(
                            "🖼️ Select ONE image",
                            matching_images,
                        )

                        st.caption(f"📍 `{selected_image}`")

                        with zipfile.ZipFile(zip_path, "r") as zip_file:
                            image_bytes = zip_file.read(selected_image)

                        image = Image.open(
                            io.BytesIO(image_bytes)
                        ).convert("RGB")

                        st.image(
                            image,
                            caption=selected_image,
                            width=700,
                        )

                        if st.button(
                            "🔍 Analyze Selected Image",
                            type="primary",
                            use_container_width=True,
                        ):
                            analyze_and_display(image)

            except Exception as error:
                st.error(f"❌ Error reading ZIP: {error}")


# ============================================================
# OPTION 3: LOCAL IMAGE PATH
# ============================================================

elif source == "📁 Select Local Image":

    st.info(
        "Enter the complete path of an image stored on "
        "your computer. This option is for local VS Code use."
    )

    local_image_path = st.text_input(
        "🖼️ Image path",
        placeholder=(
            r"C:\Users\HP\Desktop\CrowdGuard\data\videos\image.jpg"
        ),
    )

    if local_image_path:

        local_image_path = (
            local_image_path.strip().strip('"')
        )

        if not os.path.isfile(local_image_path):
            st.error("❌ Image not found. Check the path.")

        else:

            try:
                image = Image.open(
                    local_image_path
                ).convert("RGB")

                st.image(
                    image,
                    caption=os.path.basename(local_image_path),
                    width=700,
                )

                if st.button(
                    "🔍 Analyze Local Image",
                    type="primary",
                    use_container_width=True,
                ):
                    analyze_and_display(image)

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
