import math
import re
from collections import Counter

def tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

def build_index(docs):
    tokenized = []
    for doc in docs:
        tokens = tokenize(doc["title"] + " " + doc["description"])
        tokenized.append(tokens)
    return tokenized

def search(query, tokenized_docs, original_docs, k1=1.5, b=0.75):

    N = len(tokenized_docs)
    avg_dl = sum(len(d) for d in tokenized_docs) / N
    query_terms = tokenize(query)

    scores = []
    for i, doc_tokens in enumerate(tokenized_docs):
        dl = len(doc_tokens)
        tf_map = Counter(doc_tokens)
        score = 0

        for term in query_terms:
            tf = tf_map.get(term, 0)
            df = sum(1 for d in tokenized_docs if term in d)
            if df == 0:
                continue

            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm

        scores.append((score, original_docs[i]))

    return sorted(scores, key=lambda x: x[0], reverse=True)
