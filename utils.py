
# utils.py
import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

SYNONYMS = {
    "симка": ["симкарта", "сим"],
    "кредитка": ["кредитная карта", "карта"],
    "пэй": ["pay", "оплата", "пэймент"],
    "перевод": ["перевести", "отправка", "трансфер"],
}

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data3.xlsx"
]

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zа-я0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def expand_with_synonyms(query):
    words = query.split()
    expanded = set(words)
    for word in words:
        for key, synonyms in SYNONYMS.items():
            if word == key or word in synonyms:
                expanded.update([key] + synonyms)
    return list(expanded)

def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))
    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Колонки topics не найдены")
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
            print(f"Ошибка загрузки {url}: {e}")
    if not dfs:
        raise ValueError("Ни один файл не загружен")
    return pd.concat(dfs, ignore_index=True)

def semantic_search(query, df, top_k=5, threshold=0.5):
    expanded = expand_with_synonyms(preprocess(query))
    query_text = " ".join(expanded)
    query_emb = model.encode(query_text, convert_to_tensor=True)
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

def exact_match_results(query, df, min_len=8):
    q = preprocess(query)
    if len(q) < min_len:
        return []
    matched = df[df['phrase_proc'].str.contains(rf"\b{re.escape(q)}\b")]
    return list(zip(matched['phrase'], matched['topics']))
