# app.py

import streamlit as st
from utils import load_all_excels, semantic_search

# Заголовок страницы
st.set_page_config(page_title="Семантический помощник", layout="wide")
st.title("💡 Семантический ассистент")
st.markdown("Введите запрос, чтобы получить соответствующие темы из базы знаний.")

# Загрузка данных (один раз при запуске)
@st.cache_resource
def load_data():
    try:
        return load_all_excels()
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None

df = load_data()

# Ввод запроса
query = st.text_input("🔍 Ваш запрос:")

# Обработка запроса
if query and df is not None:
    with st.spinner("Анализируем..."):
        results = semantic_search(query, df, top_k=5, threshold=0.5)

        if not results:
            st.warning("❌ Ничего не найдено по вашему запросу.")
        else:
            st.markdown("### ✅ Результаты:")
            for i, (score, phrase, topics) in enumerate(results):
                highlight = "🟩" if i == 0 else "⬜️"  # выделяем самый релевантный
                st.markdown(f"""
                {highlight} **Запрос:** {phrase}  
                **Темы:** {"; ".join(topics)}  
                **Сходство:** `{score:.2f}`
                """)
