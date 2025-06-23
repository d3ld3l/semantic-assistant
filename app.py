# app.py
import streamlit as st
from utils import load_all_excels, semantic_search

st.set_page_config(page_title="Семантический помощник", layout="centered")

# Заголовок
st.title("🔍 Семантический помощник")
st.markdown("Введите фразу, и я найду связанные темы по загруженным Excel-файлам.")

# Загрузка данных (однократно)
@st.cache_data
def load_data_once():
    return load_all_excels()

try:
    df = load_data_once()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {e}")
    st.stop()

# Поле для ввода запроса
query = st.text_input("Ваш запрос:")

if query:
    with st.spinner("Ищу подходящие темы..."):
        results = semantic_search(query, df)

    if not results:
        st.warning("Ничего не найдено. Попробуйте переформулировать запрос.")
    else:
        st.markdown("### 🔎 Результаты поиска:")
        for i, (score, phrase, topics) in enumerate(results):
            bg_color = "#dff0d8" if i == 0 else "#f8f9fa"
            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                    <strong>Фраза:</strong> {phrase}<br>
                    <strong>Темы:</strong> {', '.join(topics)}<br>
                    <small><em>Сходство: {score:.2f}</em></small>
                </div>
                """,
                unsafe_allow_html=True
            )
