import streamlit as st
from PIL import Image
import torch
import os
import glob
import pandas as pd

# Fungsi untuk memuat model
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = 0.001  # Set threshold confidence
    return model

# Fungsi untuk melakukan deteksi objek
def detect_objects(model, image, save_dir="runs/detect"):
    os.makedirs(save_dir, exist_ok=True)
    results = model(image)
    results.save(save_dir=save_dir)
    return results, save_dir

# Fungsi untuk menampilkan metrik evaluasi dalam grid 2x2
def display_metrics(metrics_dir):
    metric_files = {
        "Confusion Matrix": os.path.join(metrics_dir, "confusion_matrix.png"),
        "F1 Curve": os.path.join(metrics_dir, "F1_curve.png"),
        "Precision Curve": os.path.join(metrics_dir, "P_curve.png"),
        "Recall Curve": os.path.join(metrics_dir, "R_curve.png"),
    }

    cols = st.columns(2)  # Create 2 columns for grid
    for i, (title, path) in enumerate(metric_files.items()):
        if os.path.exists(path):
            with cols[i % 2]:  # Rotate between columns
                st.image(path, caption=title, use_container_width=True)
        else:
            st.error(f"{title} not found at {path}. Please check the path.")

# Fungsi untuk merangkum hasil deteksi
def summarize_results(results, model):
    class_names = model.names  # Dictionary {class_id: class_name}
    detections = results.xyxy[0]  # Bounding box coordinates, confidence, and class ID

    if len(detections) == 0:
        return "No objects detected."

    class_counts = pd.DataFrame(detections[:, -1].tolist(), columns=["class_id"]).value_counts().to_dict()
    summary = "\n".join([f"{count} {class_names[int(class_id)]}{'s' if count > 1 else ''}"
                         for (class_id,), count in class_counts.items()])
    return summary

# Web App Streamlit
def main():
    st.header("Tugas Besar Visi Komputer Kelompok 9")
    st.text("""
    Kamelia Khoirunnisa - 1301213070
    Ashiva Prameswara   - 1301213134
    Ken Arvian Narasoma - 1301213387
    """)
    st.markdown("---")

    st.title("Trash Object Detection App")

    # Pilih model
    st.subheader("Select Model")
    model_dirs = ["model/epoch1", "model/epoch10", "model/epoch30", "model/epoch60"]
    model_options = {os.path.basename(d): os.path.join(d, "weights", "best.pt") for d in model_dirs}
    selected_model_name = st.selectbox("Choose a model:", list(model_options.keys()))
    selected_model_path = model_options[selected_model_name]

    # Tampilkan metrik evaluasi
    st.subheader("Evaluation Metrics")
    display_metrics(f"model/{selected_model_name}")

    # Upload gambar
    st.subheader("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image to upload:", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Tombol untuk mendeteksi
        if st.button("Run Detection"):
            if os.path.exists(selected_model_path):
                with st.spinner("Loading model and performing detection..."):
                    model = load_model(selected_model_path)
                    results, result_dir = detect_objects(model, image)

                # Cari file hasil deteksi
                detect_dirs = sorted(glob.glob(f"{result_dir}*"), key=os.path.getctime, reverse=True)
                if detect_dirs:
                    latest_detect_dir = detect_dirs[0]
                    result_images = glob.glob(f"{latest_detect_dir}/*.jpg")
                    if result_images:
                        result_image_path = result_images[0]
                        result_image = Image.open(result_image_path)
                        st.image(result_image, caption="Detected Objects", use_container_width=True)
                    else:
                        st.error("Detection results not found. Check the model or input image.")
                else:
                    st.error("No output directory found. Check YOLOv5 configuration.")

                # Tampilkan hasil deteksi
                st.subheader("Detection Summary")
                summary = summarize_results(results, model)
                st.text(summary)
            else:
                st.error("Model path not found. Please check the selected model.")

if __name__ == "__main__":
    main()
