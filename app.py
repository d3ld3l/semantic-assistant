# app.py
import streamlit as st
import pandas as pd
from utils import load_all_excels, semantic_search, load_model

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🧠 Semantic Assistant")

# Загрузка модели и данных
with st.spinner("Загружаем модель и данные..."):
    model = load_model()
    df = load_all_excels(model)  # Передаём модель в функцию загрузки Excel

st.markdown("---")

query = st.text_input("Введите ваш запрос")
if query:
    with st.spinner("Ищем наиболее релевантные темы..."):
        results = semantic_search(query, df, model)
        if results.empty:
            st.warning("Ничего не найдено по вашему запросу")
        else:
            st.markdown("### 🔍 Результаты:")
            for i, row in results.iterrows():
                style = "background-color:#D1FFD6; padding: 8px; border-radius: 8px;" if i == 0 else ""
                st.markdown(
                    f"<div style='{style}'><b>{row['phrase']}</b><br>Темы: {row['topics']}</div>",
                    unsafe_allow_html=True
                )
