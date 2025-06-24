# utils.py
import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data3.xlsx"
]

# Словарь синонимов
SYNONYMS = {
    "симкарта": ["сим", "симка", "сим-карта", "сим карта", "симки", "симкарты", "сим-карты", "симкарте", "сим-карту"],
    "карта": ["карточка", "картой", "карточку"],
    "утеряна": ["потеряна", "потерял", "пропала"],
    "оплатить": ["совершить оплату", "провести оплату", "произвести оплату", "осуществить оплату"],
    "не могу": ["не получается", "не проводится", "не производится", "не осуществляется", "не проходит"],
    "не проходит оплата": ["не получается оплатить", "не проводится оплата", "не производится оплата", "не осуществляется оплата", "не могу совершить оплату" , "не совершается оплата" ],
    
    "какую сумму": ["сколько"]
}

# Расширить синонимы до обратного отображения
NORMALIZED_SYNONYMS = {}
for key, values in SYNONYMS.items():
    for val in values:
        NORMALIZED_SYNONYMS[val] = key
    NORMALIZED_SYNONYMS[key] = key

# Преобразование слова к нормальной форме + синоним
def normalize_word(word):
    base = morph.parse(word)[0].normal_form
    return NORMALIZED_SYNONYMS.get(base, base)

# Нормализация текста
def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"[\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    norm_words = [normalize_word(word) for word in words]
    return " ".join(norm_words)

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
    return results[:top_k]

def exact_keyword_search(query, df):
    query_words = preprocess(query).split()
    keywords = [w for w in query_words if len(w) <= 5]
    syns = set()
    for kw in keywords:
        key = normalize_word(kw)
        syns.update([key] + [k for k, v in NORMALIZED_SYNONYMS.items() if v == key])

    matched = []
    for _, row in df.iterrows():
        words = set(row['phrase_proc'].split())
        if syns & words:
            matched.append((row['phrase'], row['topics']))
    return matched
