# utils.py
import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

# Загружаем более умную модель
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Ссылки на Excel-файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data3.xlsx"
]

# Словарь синонимов
SYNONYMS = {
    "симка": "симкарта, сим, сим-карта",
    "кредитка": "кредитная карта",
    "пэй": "pay, пей",
}

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def expand_with_synonyms(query):
    words = query.lower().split()
    expanded = set(words)
    for word in words:
        for key, synonyms in SYNONYMS.items():
            if word == key or word in synonyms.split(","):
                expanded.update(s.strip() for s in synonyms.split(","))
                expanded.add(key)
    return list(expanded)

def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))
    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Нет колонок topics")
    df = df[['phrase'] + topic_cols]
    df['topics'] = df[topic_cols].fillna('').agg(lambda x: [t for t in x.tolist() if t], axis=1)
    df['phrase_proc'] = df['phrase'].apply(preprocess)
    return df[['phrase', 'phrase_proc', 'topics']]

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

def exact_match_results(query, df, min_word_len=8):
    expanded_words = expand_with_synonyms(query)
    matches = []
    for _, row in df.iterrows():
        for word in expanded_words:
            if len(word) >= min_word_len and word in row['phrase_proc']:
                matches.append((row['phrase'], row['topics']))
                break
    return matches
