"""
Scene Cast AI: Real-Time Face Segmentation for Movie Cast Identification
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import cv2
import time
import json
import os
from datetime import datetime
from io import BytesIO

import keras
from keras import layers, Model
from keras.applications import MobileNetV2

# ─── Page Config ───
st.set_page_config(
    page_title="Scene Cast AI - Face Segmentation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMG_SIZE = 256
MODEL_PATH = os.path.join("models", "unet_face_segmentation.h5")


# ─── Custom Loss & Metrics (needed for model loading) ───
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = keras.ops.cast(keras.ops.reshape(y_true, [-1]), "float32")
    y_pred_f = keras.ops.cast(keras.ops.reshape(y_pred, [-1]), "float32")
    inter = keras.ops.sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (
        keras.ops.sum(y_true_f) + keras.ops.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = keras.ops.mean(keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = keras.ops.cast(keras.ops.reshape(y_true, [-1]), "float32")
    y_pred_f = keras.ops.cast(keras.ops.reshape(y_pred > 0.5, [-1]), "float32")
    inter = keras.ops.sum(y_true_f * y_pred_f)
    union = keras.ops.sum(y_true_f) + keras.ops.sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)


# ─── Model Builder ───
def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    skip_names = [
        "block_1_expand_relu",
        "block_3_expand_relu",
        "block_6_expand_relu",
        "block_13_expand_relu",
        "block_16_project",
    ]
    skip_outs = [base.get_layer(n).output for n in skip_names]
    enc = Model(inputs=base.input, outputs=skip_outs, name="encoder")
    enc.trainable = False

    inp = layers.Input(shape=input_shape)
    s1, s2, s3, s4, bn = enc(inp, training=False)

    def up_block(x, skip, filters):
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip])
        x = layers.SeparableConv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout2D(0.2)(x)
        x = layers.SeparableConv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        return x

    x = up_block(bn, s4, 256)
    x = up_block(x, s3, 128)
    x = up_block(x, s2, 64)
    x = up_block(x, s1, 32)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return Model(inp, out, name="UNet_MobileNetV2")


# ─── Load Model (cached) ───
@st.cache_resource
def load_model():
    model = build_unet()
    model.compile(
        optimizer="adam",
        loss=bce_dice_loss,
        metrics=[dice_coefficient, iou_metric, "accuracy"],
    )
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
    else:
        st.error(f"Model file not found: {MODEL_PATH}")
    return model


# ─── Image Processing ───
def preprocess_image(image_bytes):
    """Convert uploaded image to model input."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_rgb, img_resized, img_normalized


def predict_mask(model, img_normalized):
    """Run inference and return mask + timing."""
    inp = np.expand_dims(img_normalized, axis=0)
    start = time.time()
    pred = model.predict(inp, verbose=0)
    inference_ms = (time.time() - start) * 1000
    pred_mask = pred[0, :, :, 0]
    return pred_mask, inference_ms


def extract_faces(pred_mask, threshold=0.5, min_area=100):
    """Extract face bounding boxes from predicted mask."""
    binary = (pred_mask > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    faces = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Confidence = mean prediction value inside bounding box
        roi = pred_mask[y : y + h, x : x + w]
        confidence = float(np.mean(roi))
        faces.append(
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
             "area": int(area), "confidence": round(confidence, 4)}
        )
    return faces, binary


def draw_results(img_resized, binary_mask, faces, pred_mask):
    """Draw bounding boxes and mask overlay on image."""
    h, w = img_resized.shape[:2]
    img_display = img_resized.copy()

    # Mask overlay (green)
    overlay = img_display.copy()
    overlay[binary_mask == 1] = [0, 255, 0]
    blended = cv2.addWeighted(img_display, 0.6, overlay, 0.4, 0)

    # Bounding boxes
    img_boxes = img_resized.copy()
    for i, face in enumerate(faces):
        x, y, fw, fh = face["x"], face["y"], face["w"], face["h"]
        cv2.rectangle(img_boxes, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        label = f"Face {i+1} ({face['confidence']:.0%})"
        cv2.putText(img_boxes, label, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Heatmap
    heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return blended, img_boxes, heatmap


def create_detection_log(faces, inference_ms, image_name):
    """Create downloadable detection log."""
    log = {
        "timestamp": datetime.now().isoformat(),
        "image": image_name,
        "inference_time_ms": round(inference_ms, 2),
        "faces_detected": len(faces),
        "detections": faces,
        "model": "UNet_MobileNetV2",
        "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
    }
    return log


# ─── Streamlit UI ───
def main():
    # Sidebar
    st.sidebar.title("🎬 Scene Cast AI")
    st.sidebar.markdown("Real-Time Face Segmentation for Movie Cast Identification")
    st.sidebar.markdown("---")

    threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    min_area = st.sidebar.slider("Min Face Area (px)", 50, 500, 100, 50)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Info")
    st.sidebar.markdown("- Architecture: U-Net + MobileNetV2")
    st.sidebar.markdown("- Input: 256×256 RGB")
    st.sidebar.markdown("- Output: Binary face mask")
    st.sidebar.markdown("- Best Dice: 0.7425")

    # Main content
    st.title("🎬 Scene Cast AI")
    st.markdown("Upload a movie scene screenshot to detect and segment faces.")

    # Load model
    with st.spinner("Loading model..."):
        model = load_model()

    # Initialize session state for logs
    if "detection_logs" not in st.session_state:
        st.session_state.detection_logs = []

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        # Process image
        img_rgb, img_resized, img_normalized = preprocess_image(
            uploaded_file.getvalue()
        )

        # Run prediction
        with st.spinner("Detecting faces..."):
            pred_mask, inference_ms = predict_mask(model, img_normalized)
            faces, binary_mask = extract_faces(pred_mask, threshold, min_area)
            blended, img_boxes, heatmap = draw_results(
                img_resized, binary_mask, faces, pred_mask
            )

        # ─── Performance Dashboard ───
        st.markdown("---")
        st.subheader("📊 Performance Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Faces Detected", len(faces))
        col2.metric("Processing Time", f"{inference_ms:.1f} ms")
        avg_conf = np.mean([f["confidence"] for f in faces]) if faces else 0
        col3.metric("Avg Confidence", f"{avg_conf:.1%}")
        col4.metric("Threshold", f"{threshold:.2f}")

        # ─── Results Visualization ───
        st.markdown("---")
        st.subheader("🔍 Detection Results")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Bounding Boxes", "Segmentation Overlay", "Heatmap", "Original"]
        )

        with tab1:
            st.image(img_boxes, caption=f"{len(faces)} face(s) detected", use_container_width=True)
        with tab2:
            st.image(blended, caption="Face segmentation mask overlay", use_container_width=True)
        with tab3:
            st.image(heatmap, caption="Prediction confidence heatmap", use_container_width=True)
        with tab4:
            st.image(img_rgb, caption="Original uploaded image", use_container_width=True)

        # ─── Face Details Table ───
        if faces:
            st.markdown("---")
            st.subheader("📋 Detection Details")
            import pandas as pd

            face_df = pd.DataFrame(faces)
            face_df.index = [f"Face {i+1}" for i in range(len(faces))]
            face_df.columns = ["X", "Y", "Width", "Height", "Area (px)", "Confidence"]
            st.dataframe(face_df, use_container_width=True)

        # ─── Download & Export ───
        st.markdown("---")
        st.subheader("💾 Download & Export")

        log = create_detection_log(faces, inference_ms, uploaded_file.name)
        st.session_state.detection_logs.append(log)

        col_dl1, col_dl2, col_dl3 = st.columns(3)

        # Download detection log (JSON)
        with col_dl1:
            log_json = json.dumps(log, indent=2)
            st.download_button(
                "📄 Download Detection Log (JSON)",
                data=log_json,
                file_name=f"detection_log_{uploaded_file.name}.json",
                mime="application/json",
            )

        # Download mask image
        with col_dl2:
            mask_img = (binary_mask * 255).astype(np.uint8)
            _, mask_buf = cv2.imencode(".png", mask_img)
            st.download_button(
                "🎭 Download Face Mask (PNG)",
                data=mask_buf.tobytes(),
                file_name=f"mask_{uploaded_file.name}.png",
                mime="image/png",
            )

        # Download annotated image
        with col_dl3:
            boxes_bgr = cv2.cvtColor(img_boxes, cv2.COLOR_RGB2BGR)
            _, boxes_buf = cv2.imencode(".jpg", boxes_bgr)
            st.download_button(
                "📸 Download Annotated Image",
                data=boxes_buf.tobytes(),
                file_name=f"annotated_{uploaded_file.name}",
                mime="image/jpeg",
            )

        # ─── Session Detection History ───
        if len(st.session_state.detection_logs) > 1:
            st.markdown("---")
            st.subheader("📜 Detection History (this session)")
            history_json = json.dumps(st.session_state.detection_logs, indent=2)
            st.download_button(
                "📥 Download All Logs",
                data=history_json,
                file_name="all_detection_logs.json",
                mime="application/json",
            )
            for i, entry in enumerate(reversed(st.session_state.detection_logs)):
                st.text(
                    f"{i+1}. {entry['image']} | "
                    f"{entry['faces_detected']} faces | "
                    f"{entry['inference_time_ms']:.1f}ms"
                )

    else:
        # Show sample when no image uploaded
        st.info("👆 Upload a movie scene screenshot to get started.")
        sample_path = os.path.join("data", "Part 1Test Data - Prediction Image.jpeg")
        if os.path.exists(sample_path):
            st.markdown("Or try with the sample test image:")
            if st.button("🎬 Run on Sample Image"):
                with open(sample_path, "rb") as f:
                    sample_bytes = f.read()
                img_rgb, img_resized, img_normalized = preprocess_image(sample_bytes)
                pred_mask, inference_ms = predict_mask(model, img_normalized)
                faces, binary_mask = extract_faces(pred_mask, threshold, min_area)
                blended, img_boxes, heatmap = draw_results(
                    img_resized, binary_mask, faces, pred_mask
                )
                st.image(img_boxes, caption=f"{len(faces)} face(s) detected", use_container_width=True)
                st.metric("Processing Time", f"{inference_ms:.1f} ms")


if __name__ == "__main__":
    main()
