import streamlit as st
from PIL import Image
from ..inference.predict import load_model, predict_image

st.set_page_config(page_title="Track Detector", page_icon="🧠", layout="centered")

st.title("🧭 Распознавание следов")
st.write("Загрузите изображение, и модель определит класс следа.")

MODEL_PATH = "model/model_best.pth"
CLASS_NAMES = ['Bear', 'Bird', 'Cat', 'Dog', 'Leopard', 'Otter']

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

uploaded_file = st.file_uploader("📸 Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    if st.button("🔍 Определить класс"):
        with st.spinner("Анализ изображения..."):
            label, probs = predict_image(model, image, CLASS_NAMES)

        st.success(f"✅ Предсказанный класс: {label}")

        if st.button("📊 Показать вероятности всех классов"):
            st.bar_chart(probs, x=CLASS_NAMES)