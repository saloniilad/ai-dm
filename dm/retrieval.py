"""Hybrid retrieval: BM25 (+ optional dense embeddings fused with RRF) with a geospatial
proximity bonus and an optional cross-encoder reranker."""
import math
import re
import sys
from collections import Counter

from .geo import haversine_km

STOP = set("""a an the of to in on at is was were and or for with by from as it its this that his her their be been
are did do does what who which how when where why had has have i you he she they we me my your not but if then so""".split())


def _stem(t):
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def tokenize(text):
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())
            if (len(t) > 1 or t.isdigit()) and t not in STOP]


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(doc) for doc in corpus]
        self.dl = [len(doc) for doc in corpus]
        self.N = len(corpus)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 1.0
        self.df = Counter()
        for c in self.tf:
            self.df.update(c.keys())

    def scores(self, query_tokens):
        out = [0.0] * self.N
        for term in set(query_tokens):
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return out


def _ranks(scores, top=None, positive_only=True):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    if positive_only:
        order = [i for i in order if scores[i] > 0]
    if top:
        order = order[:top]
    return {i: r for r, i in enumerate(order)}


def _doc_text(d):
    return d["title"] + ". " + d["text"]


class DenseIndex:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.cache = {}

    def scores(self, query, docs):
        import numpy as np
        missing = [d for d in docs if d["id"] not in self.cache]
        if missing:
            embs = self.model.encode([_doc_text(d) for d in missing], normalize_embeddings=True)
            for d, e in zip(missing, embs):
                self.cache[d["id"]] = e
        mat = np.stack([self.cache[d["id"]] for d in docs])
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return (mat @ q).tolist()


class Retriever:
    def __init__(self, dense=False, rerank=False, geo_weight=0.25, geo_scale_km=10.0,
                 dense_model="BAAI/bge-small-en-v1.5", rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.geo_weight, self.geo_scale_km = geo_weight, geo_scale_km
        self.dense = self.reranker = None
        self._key, self._bm = None, None
        if dense:
            try:
                self.dense = DenseIndex(dense_model)
            except Exception as e:  # missing package / no network
                print(f"[retrieval] dense disabled: {e}", file=sys.stderr)
        if rerank:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(rerank_model)
            except Exception as e:
                print(f"[retrieval] reranker disabled: {e}", file=sys.stderr)

    def _bm25(self, docs):
        key = tuple(d["id"] for d in docs)
        if key != self._key:
            self._bm, self._key = BM25([tokenize(_doc_text(d)) for d in docs]), key
        return self._bm

    def search(self, query, docs, k=4, near=None, coords_of=None):
        if not docs:
            return []
        bm = self._bm25(docs).scores(tokenize(query))
        if self.dense:
            r_bm, r_de = _ranks(bm), _ranks(self.dense.scores(query, docs), top=30, positive_only=False)
            base = [(1 / (60 + r_bm[i]) if i in r_bm else 0) + (1 / (60 + r_de[i]) if i in r_de else 0)
                    for i in range(len(docs))]
        else:
            base = list(bm)
        mx = max(base)
        if mx <= 0:
            return []
        base = [s / mx for s in base]
        if near and coords_of and self.geo_weight:
            for i, d in enumerate(docs):
                if base[i] > 0:
                    c = coords_of(d)
                    if c:
                        base[i] += self.geo_weight * math.exp(-haversine_km(near, c) / self.geo_scale_km)
        order = [i for i in sorted(range(len(docs)), key=lambda i: -base[i]) if base[i] > 0]
        if self.reranker:
            cand = order[:max(k, 10)]
            rs = self.reranker.predict([(query, _doc_text(docs[i])) for i in cand])
            order = [i for _, i in sorted(zip(rs, cand), key=lambda x: -x[0])]
        return [docs[i] for i in order[:k]]
