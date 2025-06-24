# app.py
import streamlit as st
from utils import load_all_excels, semantic_search

st.set_page_config(page_title="Семантический помощник", layout="wide")
st.title("🔍 Семантический помощник")

with st.spinner("Загрузка данных..."):
    try:
        df = load_all_excels()
        st.success("Данные успешно загружены")
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        st.stop()

query = st.text_input("Введите фразу для поиска:")

if query:
    results = semantic_search(query, df)

    if results:
        st.markdown("### 🔎 Результаты поиска:")
        for idx, (score, phrase, topics) in enumerate(results):
            color = "#D1F2EB" if idx == 0 else "#F4F6F7"
            st.markdown(
                f"<div style='background-color:{color}; padding:10px; border-radius:10px; margin-bottom:10px;'>"
                f"<strong>{phrase}</strong><br><small>Темы:</small> {', '.join(topics)}<br>"
                f"<small>Схожесть: {score:.2f}</small></div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("Совпадений не найдено.")
