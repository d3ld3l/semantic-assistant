# utils.py
import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import json

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

# Загрузка словаря синонимов из файла
try:
    with open("synonyms.json", "r", encoding="utf-8") as f:
        SYNONYM_GROUPS = json.load(f)
except Exception as e:
    print(f"Ошибка загрузки синонимов: {e}")
    SYNONYM_GROUPS = []

# Построим карту синонимов
SYNONYM_MAP = {}
for group in SYNONYM_GROUPS:
    for word in group:
        SYNONYM_MAP[word] = group[0]  # Все ссылаются на первый элемент

# Обработка текста с учетом синонимов
def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"[-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()
    norm_tokens = [SYNONYM_MAP.get(token, token) for token in tokens]
    return " ".join(norm_tokens)

def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))

    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

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

    # Точный поиск по коротким словам <= 5 символов
    if len(query.strip()) <= 5:
        matches = df[df['phrase_proc'].str.contains(rf"\\b{re.escape(query.strip().lower())}\\b")]
        for _, row in matches.iterrows():
            phrase, topics = row['phrase'], row['topics']
            if (1.0, phrase, topics) not in results:
                results.append((1.0, phrase, topics))

    return results[:top_k]
