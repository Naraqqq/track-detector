import sys, os
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import time

# ==== Импорты из проекта ====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.predict import load_model, predict_image

# ==== Настройки страницы ====
st.set_page_config(page_title="Track Detector", page_icon="🧭", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #f9fafc;
        font-family: "Segoe UI", sans-serif;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #0078ff;
        color: white;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #005edb;
        transform: scale(1.02);
    }
    .fade-in {
        animation: fadeIn 1.2s ease-in-out;
    }
    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(10px);}
        100% {opacity: 1; transform: translateY(0);}
    }
    .title {
        text-align: center;
        color: #1f1f1f;
    }
    .footer {
        text-align: center;
        color: #777;
        margin-top: 40px;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# ==== Константы ====
MODEL_PATH = "model/efficientnet_model.pth"
CLASS_NAMES = ['Bear', 'Bird', 'Cat', 'Dog', 'Leopard', 'Otter']


# ==== Кэшируем модель ====
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


model = get_model()

# ==== Навигация ====
st.sidebar.title("🔧 Навигация")
page = st.sidebar.radio("Выберите раздел:", ["🎯 Предсказание", "📊 Аналитика", "ℹ️ О модели"])

# ==== Состояние ====
if "history" not in st.session_state:
    st.session_state["history"] = []

if page == "🎯 Предсказание":
    st.markdown("<h2 class='title'>🧭 Распознавание следов животных</h2>", unsafe_allow_html=True)
    st.write("Загрузите одно или несколько изображений, и модель определит, какому животному принадлежат следы.")

    uploaded_files = st.file_uploader(
        "📸 Загрузите изображения",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🔍 Определить классы для всех изображений"):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                with st.spinner(f"Анализ {uploaded_file.name}..."):
                    time.sleep(0.5)
                    label, probs = predict_image(model, image, CLASS_NAMES)
                color = "green"

                st.markdown(f"""
                <div class='fade-in' style='
                    border: 1px solid #ddd;
                    border-radius: 15px;
                    padding: 10px 20px;
                    background-color: #fff;
                    margin-bottom: 20px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
                    transition: all 0.3s ease-in-out;
                '>
                    <h4 style='color:{color}; margin-bottom:5px;'>📷 {uploaded_file.name}</h4>
                    <p style='font-size:1.1em;'>Предсказанный класс: <b>{label}</b>)</p>
                </div>
                """, unsafe_allow_html=True)

                st.image(image, caption=f"{label}", use_container_width=True)

                # График вероятностей
                fig = px.bar(
                    x=CLASS_NAMES, y=probs, color=CLASS_NAMES,
                    title=f"Распределение вероятностей для {uploaded_file.name}",
                    labels={"x": "Класс", "y": "Вероятность"}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Добавление в историю
                st.session_state["history"].append({
                    "filename": uploaded_file.name,
                    "label": label,
                    "probs": probs
                })

        if st.button("🗑 Очистить историю"):
            st.session_state["history"] = []


elif page == "📊 Аналитика":
    st.markdown("<h2 class='title'>📊 Аналитика предсказаний</h2>", unsafe_allow_html=True)

    if not st.session_state["history"]:
        st.info("Нет данных для анализа. Сделайте хотя бы одно предсказание.")
    else:
        df = pd.DataFrame(st.session_state["history"])
        summary = df["label"].value_counts().reset_index()
        summary.columns = ["Класс", "Количество"]

        st.subheader("📈 Частота предсказанных классов")
        fig = px.bar(summary, x="Класс", y="Количество", color="Класс", text="Количество")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🕓 История последних 5 предсказаний")
        for item in st.session_state["history"][-5:][::-1]:
            st.write(f"📷 {item['filename']} → {item['label']}")

elif page == "ℹ️ О модели":
    st.markdown("<h2 class='title'>ℹ️ Информация о модели</h2>", unsafe_allow_html=True)

    st.write("""
    Модель: EfficientNet-B4, обученная на своих данных и со своей архитектурой

    Классы: 6 (Bear, Bird, Cat, Dog, Leopard, Otter)  

    Источник весов: [🤗 Hugging Face](https://huggingface.co/VladimirFireBall/efficientnet-steps)
    """)

    st.markdown("<div class='footer'>© 2025 Track Detector | Разработано командой ptp2025</div>",
                unsafe_allow_html=True)