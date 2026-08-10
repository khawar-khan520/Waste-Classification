"""
app.py - Interactive Streamlit app for the Waste Classification project.

Upload a checkpoint (trained via src/train.py) and an image, and get:
  - predicted class + confidence
  - full probability breakdown
  - a Grad-CAM heatmap showing what the model focused on

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import get_model  # noqa: E402

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def get_target_layer(model, model_name):
    if model_name == 'resnet50':
        return [model.layer4[-1]]
    elif model_name == 'efficientnet':
        return [model.conv_head]
    elif model_name == 'vit':
        return [model.blocks[-1].norm1]


@st.cache_resource
def load_model(model_name: str, checkpoint_bytes: bytes, num_classes: int):
    """Load a model + checkpoint. Cached so re-running the app doesn't reload every time."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_classes=num_classes, pretrained=False)

    import io
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {}
    if isinstance(checkpoint, dict):
        meta['epoch'] = checkpoint.get('epoch', 'unknown')
        meta['val_balanced_acc'] = checkpoint.get('val_balanced_acc', None)

    return model, device, meta


def predict(model, device, image: Image.Image):
    tensor = EVAL_TRANSFORM(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs, tensor


def make_gradcam(model, model_name, tensor, pred_idx, device, original_image: Image.Image):
    target_layers = get_target_layer(model, model_name)
    reshape_transform = vit_reshape_transform if model_name == 'vit' else None

    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    # Prepare the (resized/cropped) RGB image in [0,1] to overlay the heatmap on
    display_img = original_image.convert('RGB').resize((224, 224))
    rgb_float = np.array(display_img).astype(np.float32) / 255.0

    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return visualization


def main():
    st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")
    st.title("♻️ Waste Classification")
    st.caption("Interactive demo for the waste-classification domain generalization project (University of Bamberg).")

    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox(
            "Architecture (must match your checkpoint)",
            options=['resnet50', 'efficientnet', 'vit'],
            format_func=lambda x: {'resnet50': 'ResNet-50', 'efficientnet': 'EfficientNet-B3', 'vit': 'ViT-Small/16'}[x],
        )
        num_classes = st.radio("Number of classes", options=[6, 5], index=0,
                                help="6-class baseline, or 5-class border-background-aug variant from the report.")
        checkpoint_file = st.file_uploader("Checkpoint file (.pth / .pt)", type=["pth", "pt"])
        show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True, disabled=not GRADCAM_AVAILABLE)
        if not GRADCAM_AVAILABLE:
            st.caption("Install `grad-cam` (in requirements.txt) to enable this.")

    if checkpoint_file is None:
        st.info("👈 Upload a trained checkpoint (.pth) in the sidebar to get started. "
                 "This should be a file saved by `src/train.py`, e.g. `checkpoints/resnet50_best.pth`.")
        return

    class_names = CLASS_NAMES if num_classes == 6 else [f"class_{i}" for i in range(5)]

    with st.spinner("Loading model..."):
        try:
            model, device, meta = load_model(model_name, checkpoint_file.getvalue(), num_classes)
        except Exception as e:
            st.error(f"Couldn't load checkpoint with architecture '{model_name}' / {num_classes} classes: {e}")
            st.stop()

    if meta:
        cols = st.columns(2)
        if meta.get('epoch') is not None:
            cols[0].metric("Checkpoint epoch", meta['epoch'])
        if meta.get('val_balanced_acc') is not None:
            cols[1].metric("Val. balanced acc.", f"{meta['val_balanced_acc']:.1%}")

    st.divider()
    uploaded_image = st.file_uploader("Upload an image of waste to classify", type=["jpg", "jpeg", "png"])

    if uploaded_image is None:
        st.info("Upload a JPG/PNG image above to classify it.")
        return

    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)
    col1.image(image, caption="Input image", use_container_width=True)

    probs, tensor = predict(model, device, image)
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    with col2:
        st.subheader(f"Prediction: **{pred_label}**")
        st.caption(f"Confidence: {probs[pred_idx]:.1%}")
        st.bar_chart({class_names[i]: float(probs[i]) for i in range(len(probs))})

    if show_gradcam and GRADCAM_AVAILABLE:
        st.divider()
        st.subheader("Grad-CAM — what the model focused on")
        try:
            with st.spinner("Generating Grad-CAM..."):
                heatmap = make_gradcam(model, model_name, tensor, pred_idx, device, image)
            st.image(heatmap, caption="Warmer colors = regions the model relied on most", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM failed for this architecture/checkpoint: {e}")


if __name__ == "__main__":
    main()
