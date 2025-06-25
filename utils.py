# Обновлённый utils.py с учётом разделения фраз по слэшу и учётом синонимов для точного поиска
import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

# Словарь взаимных синонимов
SYNONYM_GROUPS = [
    ["симка", "сим-карта", "сим карта", "сим", "симкарта", "сим-карты", "симкарты", "симке", "симки", "сим-карту"],
    ["кредитка", "кредитная карта"],
    ["наличные", "наличка"]
]

# Создание словаря замен
SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    for word in group:
        SYNONYM_DICT[word] = group[0]  # Назначаем первое как базовое значение

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Заменяем синонимы на базовые слова
    for key, val in SYNONYM_DICT.items():
        pattern = r"\\b" + re.escape(key) + r"\\b"
        text = re.sub(pattern, val, text)
    return text

def split_slash_phrases(phrase):
    return [p.strip() for p in str(phrase).split("/") if p.strip()]

def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))

    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    phrases = []
    for _, row in df.iterrows():
        raw_phrases = split_slash_phrases(row['phrase'])
        topics = [t for t in row[topic_cols].fillna('').tolist() if t]
        for phrase in raw_phrases:
            phrases.append({
                'phrase': phrase,
                'phrase_proc': preprocess(phrase),
                'topics': topics
            })

    return pd.DataFrame(phrases)

def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            df = load_excel(url)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

def semantic_search(query, df, top_k=5, threshold=0.5):
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    phrase_embs = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)

    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]
    results = []

    for idx, score in enumerate(sims):
        score = float(score)
        if score >= threshold:
            phrase = df.iloc[idx]['phrase']
            topics = df.iloc[idx]['topics']
            results.append((score, phrase, topics))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def exact_keyword_search(query, df, max_len=5):
    words = preprocess(query).split()
    found = set()
    matches = []

    for word in words:
        if len(word) > max_len:
            continue
        # Приводим к базовой форме по словарю синонимов
        base = SYNONYM_DICT.get(word, word)
        for idx, row in df.iterrows():
            if base in row['phrase_proc']:
                key = (row['phrase'], tuple(row['topics']))
                if key not in found:
                    matches.append((row['phrase'], row['topics']))
                    found.add(key)
    return matches
