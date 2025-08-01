import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Streamlit page setup
st.set_page_config(page_title="GenAI Image Generator", layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png", width=150)
st.sidebar.title("🧠 GenAI Controls")
guidance = st.sidebar.slider("🎛️ Guidance Scale", 1.0, 20.0, 7.5, step=0.5)
height = st.sidebar.selectbox("📐 Image Height", [512, 640, 768], index=0)
width = st.sidebar.selectbox("📏 Image Width", [512, 640, 768], index=0)

# Title and input
st.markdown("<h1 style='text-align:center;'>🎨 Generate AI Images</h1>", unsafe_allow_html=True)
prompt = st.text_input("Enter your prompt:", placeholder="A futuristic city floating in the sky 🌇")

generate = st.button("🚀 Generate Image")

# Load pipeline (cached)
@st.cache_resource
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")

pipe = load_pipeline()

# Generate image
if generate and prompt:
    with st.spinner("Generating image... please wait"):
        image = pipe(prompt, guidance_scale=guidance, height=height, width=width).images[0]
        st.image(image, caption=prompt, use_container_width=True)

        # Download button
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button("⬇️ Download Image", buf.getvalue(), file_name="genai_image.png", mime="image/png")