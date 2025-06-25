import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Ссылки на Excel-файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data3.xlsx"
]

# Универсальный словарь синонимов (все синонимы эквивалентны друг другу)
SYNONYMS = [
    ["сим", "симка", "сим-карта", "сим карта", "симкарта", "симки", "симкарты", "сим-карты", "симкарте", "сим-карту"],
    ["карта", "карточка", "картой", "карточку"],
    ["кредитка", "кредитная карта"],
    ["наличные", "наличка"],
    ["не проходит", "не проводится", "не осуществляется", "не производится", "не получается"],
    ["оплатить", "совершить оплату", "провести оплату", "произвести оплату", "осуществить оплату"],
]

# Создание синонимичного словаря (каждое слово → его каноническая форма)
def build_synonym_map(synonym_groups):
    mapping = {}
    for group in synonym_groups:
        canonical = group[0]
        for word in group:
            mapping[word] = canonical
    return mapping

SYNONYM_MAP = build_synonym_map(SYNONYMS)

# Функция нормализации и замены синонимов
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Замена по словарю синонимов
    words = text.split()
    normalized = []
    for word in words:
        base = word
        for suffix in ["у", "е", "и", "а", "ы", "о", "ю", "е", "ой"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        replaced = SYNONYM_MAP.get(word, SYNONYM_MAP.get(base, word))
        normalized.append(replaced)
    return " ".join(normalized)

# Загрузка Excel-файлов
def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))

    # Деление фраз по "/"
    df = df[df['phrase'].notna()]
    df = df.assign(phrase=df['phrase'].astype(str))
    df = df.assign(phrase=df['phrase'].str.split(r"\s*/\s*")).explode("phrase")

    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df['topics'] = df[topic_cols].fillna('').agg(lambda x: [t for t in x.tolist() if t], axis=1)
    df['phrase_proc'] = df['phrase'].apply(preprocess)
    return df[['phrase', 'phrase_proc', 'topics']]

# Загрузка всех Excel-файлов
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

# Семантический поиск
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

# Ключевой точный поиск (по слову до 5 символов)
def exact_keyword_search(query, df, max_len=5):
    query_proc = preprocess(query)
    key_words = [w for w in query_proc.split() if len(w) <= max_len]
    matches = df[df['phrase_proc'].apply(lambda p: any(k in p for k in key_words))]
    return matches[['phrase', 'topics']].drop_duplicates().values.tolist()
