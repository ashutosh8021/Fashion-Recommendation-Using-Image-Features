import streamlit as st
from PIL import Image
from recommender import FashionRecommender
import os

# Download large files if not present
if not os.path.exists("models/faiss_index.bin"):
    st.write("Downloading model files from Google Drive...")
    os.system("python download_model_files.py")

@st.cache_resource(show_spinner="Loading model & index …")
def load_sys():
    return FashionRecommender()

recomm = load_sys()

st.set_page_config(page_title="Fashion Image Recommender")
st.title("👚 Fashion Visual Recommender")
st.write("Upload a garment photo and I’ll show visually similar items.")

file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Query", width=250)
    with st.spinner("Searching ..."):
        hits = recomm.search(img)
    cols = st.columns(len(hits))
    for col, p in zip(cols, hits):
        col.image(Image.open(p), use_column_width=True)
