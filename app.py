# app.py
import streamlit as st
from utils import load_model, load_data, semantic_search

st.set_page_config(page_title="Semantic Assistant", layout="wide")
st.title("🤖 Семантический ассистент")

model = load_model()
df = load_data(model)

query = st.text_input("Введите ваш запрос:")

if query:
    results, extras = semantic_search(query, df, model)

    if results:
        st.markdown("### 🎯 Наиболее релевантные результаты:")
        for score, phrase, topics in results:
            st.markdown(f"<div style='background-color:#e0ffe0; padding:10px; border-radius:10px;'><b>{phrase}</b><br><small>Темы: {', '.join(topics)}</small><br><small>Сходство: {score:.2f}</small></div><br>", unsafe_allow_html=True)

    if extras:
        st.markdown("### 🔍 Прямые вхождения по коротким словам:")
        for phrase, topics in extras:
            st.markdown(f"<div style='background-color:#f0f0f0; padding:10px; border-radius:10px;'><b>{phrase}</b><br><small>Темы: {', '.join(topics)}</small></div><br>", unsafe_allow_html=True)
    
    if not results and not extras:
        st.warning("Ничего не найдено по вашему запросу.")
