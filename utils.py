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

# Словарь синонимов (взаимные)
SYNONYM_GROUPS = [
    ["сим", "симка", "сим-карта", "сим карта", "симкарта", "симки", "симкарты", "сим-карты", "симкарте", "сим-карту"],
    ["карта", "карточка", "картой", "карточку"],
    ["наличные", "наличка"],
    ["кредитка", "кредитная карта"],
    ["оплатить", "совершить оплату", "провести оплату", "произвести оплату", "осуществить оплату"],
]

# Расширим до словарь: любое слово → его базовая форма
SYNONYM_MAP = {}
for group in SYNONYM_GROUPS:
    for word in group:
        for synonym in group:
            SYNONYM_MAP[synonym] = group[0]

def apply_synonyms(text):
    words = text.split()
    return " ".join([SYNONYM_MAP.get(word, word) for word in words])

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = apply_synonyms(text)
    return text

def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))

    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df = df[['phrase'] + topic_cols]
    df['topics'] = df[topic_cols].fillna('').agg(lambda x: [t for t in x if pd.notna(t)], axis=1)
    df['phrase_proc'] = df['phrase'].apply(lambda x: preprocess(str(x)))
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

def keyword_search(query, df):
    query_words = preprocess(query).split()
    short_words = [word for word in query_words if len(word) <= 5]
    matched = []
    for _, row in df.iterrows():
        phrase_words = row['phrase_proc'].split()
        if any(word in phrase_words for word in short_words):
            matched.append((row['phrase'], row['topics']))
    return matched
