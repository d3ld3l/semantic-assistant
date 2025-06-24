# app.py
import streamlit as st
from utils import load_all_excels, semantic_search, exact_word_match, highlight_answer
import time

st.set_page_config(page_title="Семантический помощник", layout="wide")

# Загрузка данных
with st.spinner("Загрузка базы данных..."):
    try:
        df = load_all_excels()
        st.success("✅ Данные успешно загружены!")
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        st.stop()

# Заголовок
st.title("🤖 Семантический помощник")
st.markdown("Введите фразу, и помощник найдет соответствующие темы:")

# Ввод пользователем
query = st.text_input("Ваш запрос:")

if query:
    with st.spinner("🔎 Выполняется поиск..."):
        time.sleep(0.5)

        # Основной семантический поиск
        semantic_results = semantic_search(query, df)

        # Точный поиск по короткому слову (до 8 символов)
        extra_matches = exact_word_match(query, df, max_len=8)

        # Объединение результатов и удаление дубликатов
        phrases_seen = set()
        all_results = []

        for score, phrase, topics in semantic_results + extra_matches:
            if phrase not in phrases_seen:
                all_results.append((score, phrase, topics))
                phrases_seen.add(phrase)

        if all_results:
            st.markdown("### 🔍 Результаты поиска:")
            for score, phrase, topics in all_results:
                highlighted = highlight_answer(phrase, query)
                st.markdown(f"<div style='padding:8px;margin-bottom:6px;border-radius:8px;background-color:#f5f5f5'><strong>{highlighted}</strong><br/><em>Темы:</em> {', '.join(topics)}</div>", unsafe_allow_html=True)
        else:
            st.warning("Не найдено релевантных результатов.")
