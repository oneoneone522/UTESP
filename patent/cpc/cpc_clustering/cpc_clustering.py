# -*- coding: utf-8 -*-
"""
cpc_clustering.py -- 對應論文 4 節 / Fig.2 的 "Topic Generation" 步驟。

流程:
  1. 讀 cpc_phrase.txt,每行一個 CPC 短語
  2. 用已訓練好的 skip-gram 向量(skipgram.py 的 WordVectors)算每個短語的向量
  3. 對這些向量做階層式聚類(HAC),論文只說 "minimum cluster size is 3",
     沒有給完整演算法,這裡的做法是我的設計選擇,不是論文原文:
       (a) 用 average-linkage + cosine distance 建聚類樹
       (b) 從一個起始群數開始切(--n_clusters),得到初始分群
       (c) 把任何小於 min_cluster_size 的群,合併進離它(以群心 cosine 距離)
           最近的「合法」群,重複直到所有群都 >= min_cluster_size
  4. 每個群的中心(群內短語向量的平均)就是一個 Topic,
     存成 {topic_id: 中心向量} 和 {topic_id: 群內短語清單},供後續
     candidate score 的 Topic_relation(公式 3)使用。

用法:
    python3 cpc_clustering.py --selftest
    python3 cpc_clustering.py \
        --cpc_phrase_file example_data/example_cpc/cpc_phrase.txt \
        --model_prefix models/sg_full \
        --out patent/cpc/cpc_clustering/cpc_topics \
        --n_clusters 20 --min_cluster_size 3
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.abspath(os.path.join(_here, '..', '..')))  # patent/,skipgram.py 在這裡
from skipgram import WordVectors


# Step 1: 讀檔案，算向量----------------------------------------------------------------------------

def load_cpc_phrases(path):
    phrases = []
    seen = set()
    with open(path, encoding='utf-8') as file:
        for line in file:
            p = line.strip().lower()
            if p and p not in seen:
                phrases.append(p)
                seen.add(p)
        return phrases


def embed_phrases(phrases, wv):
    kept, vecs, oov = [], [], []
    for p in phrases:
        v = wv.phrase_vector(p)
        if v is None:
            oov.append(p)
        else:
            kept.append(p)
            vecs.append(v)
    return kept, np.array(vecs, dtype=np.float64), oov



# Step 2: 聚類 ----------------------------------------------------------------------------
def hac_with_min_size(vecs, n_clusters, min_cluster_size, linkage_method='average'):
    """階層式聚類,並強制每個群至少有 min_cluster_size 個成員。

    回傳每個短語的群編號(labels,從 0 開始連續編號)。
    """
    n = len(vecs)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n <= min_cluster_size:
        # 短語太少,全部當同一群,否則湊不出一個合法的群
        return np.zeros(n, dtype=int)

    n_clusters = min(n_clusters, n)
    dist = pdist(vecs, metric='cosine')
    dist = np.nan_to_num(dist, nan=2.0)  # 零向量等退化情況,距離設為最大值 2
    Z = linkage(dist, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1  # 轉成 0-based

    # 強制最小群大小:每輪找出「目前最小的那一群」,把它併進離它群心
    # 最近的「另一群」(不限制對方是否已達標)。這樣兩個小群可以先彼此合併,
    # 不會被迫立刻擠進最大的那一群(最早的版本就是這樣,導致整個聚類塌縮成一群)。
    def centroid(lab, k):
        return vecs[lab == k].mean(axis=0)

    labels = labels.copy()
    while True:
        uniq = sorted(set(labels.tolist()))
        sizes = {k: int((labels == k).sum()) for k in uniq}
        if len(uniq) <= 1 or min(sizes.values()) >= min_cluster_size:
            break
        k_min = min(sizes, key=lambda k: sizes[k])
        c_min = centroid(labels, k_min)
        others = [k for k in uniq if k != k_min]
        centroids = {k: centroid(labels, k) for k in others}
        best = min(others, key=lambda j: 1 - c_min @ centroids[j] /
                   (np.linalg.norm(c_min) * np.linalg.norm(centroids[j]) + 1e-12))
        labels[labels == k_min] = best

    # 重新編號成連續的 0..K-1
    uniq = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(uniq)}
    return np.array([remap[v] for v in labels])


def build_topics(phrases, vecs, labels):
    topics = {}
    for k in sorted(set(labels.tolist())):
        idx = np.where(labels == k)[0]
        topics[k] = {
            'phrases': [phrases[i] for i in idx],
            'centroid': vecs[idx].mean(axis=0),
        }
    return topics


def save_topics(topics, out_prefix):
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    ids = sorted(topics.keys())
    centroids = np.stack([topics[i]['centroid'] for i in ids])
    np.save(out_prefix + '_centroids.npy', centroids)
    with open(out_prefix + '_phrases.json', 'w', encoding='utf-8') as f:
        json.dump({str(i): topics[i]['phrases'] for i in ids}, f, ensure_ascii=False, indent=2)


class TopicModel:
    """下游(Graph Construction)拿這個來算 Topic_relation(論文公式 3)。"""

    def __init__(self, centroids):
        self.centroids = np.asarray(centroids, dtype=np.float64)
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self._unit = self.centroids / np.clip(norms, 1e-12, None)

    @classmethod
    def load(cls, out_prefix):
        return cls(np.load(out_prefix + '_centroids.npy'))

    def topic_relation(self, phrase_vec):
        """公式(3): Topic_relation_i = max_k cos(theta_i, Topic_k)。"""
        v = phrase_vec / (np.linalg.norm(phrase_vec) + 1e-12)
        return float(np.max(self._unit @ v))


# ----------------------------------------------------------------------------
def selftest():
    """合成有明確分群結構的向量,檢查聚類是否能還原分群,且沒有群小於 min_size。"""
    rng = np.random.default_rng(0)
    K, per, dim = 6, 8, 16
    true_label = np.repeat(np.arange(K), per)
    centers = rng.normal(size=(K, dim)) * 3
    vecs = centers[true_label] + rng.normal(size=(K * per, dim)) * 0.3
    phrases = ['p%03d' % i for i in range(len(vecs))]

    labels = hac_with_min_size(vecs, n_clusters=K, min_cluster_size=3)
    sizes = np.bincount(labels)
    print('群數: %d, 各群大小: %s' % (len(sizes), sizes.tolist()))
    ok_size = all(s >= 3 for s in sizes)
    print('每群都 >= 3 個成員: %s' % ('通過' if ok_size else '失敗'))

    # 用 adjusted purity 概念檢查:同一個真實主題的點,是不是大多被分到同一群
    agree = 0
    for k in range(K):
        idx = np.where(true_label == k)[0]
        vals, counts = np.unique(labels[idx], return_counts=True)
        agree += counts.max()
    purity = agree / len(vecs)
    print('分群純度(越接近1越好): %.3f  %s' % (purity, '通過' if purity > 0.9 else '失敗'))

    topics = build_topics(phrases, vecs, labels)
    tm = TopicModel(np.stack([topics[i]['centroid'] for i in sorted(topics)]))
    # 同一真實主題裡的點,topic_relation 應該要比跨主題的點更高
    same_topic_sims, diff_topic_sims = [], []
    for i in range(0, len(vecs), per):
        same_topic_sims.append(tm.topic_relation(vecs[i]))
    rel = [tm.topic_relation(v) for v in vecs]
    print('topic_relation 範圍: min=%.3f max=%.3f(應接近 1,因為每個點附近就有自己的群心)'
          % (min(rel), max(rel)))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='CPC 短語聚類,產生 Topic Generation 的輸出')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--cpc_phrase_file')
    ap.add_argument('--model_prefix', help='skipgram.py 訓練輸出的前綴,例如 models/sg_full')
    ap.add_argument('--out', help='輸出前綴,例如 patent/cpc/cpc_clustering/cpc_topics')
    ap.add_argument('--n_clusters', type=int, default=20,
                     help='初始切幾群(論文沒給,先設一個預設值,可調)')
    ap.add_argument('--min_cluster_size', type=int, default=3,
                     help='論文 4.1 節:CPC guiding phrase 的 minimum cluster size = 3')
    ap.add_argument('--linkage', default='average', choices=['average', 'complete', 'single'])
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    for name in ('cpc_phrase_file', 'model_prefix', 'out'):
        if getattr(a, name) is None:
            ap.error('需要 --%s(或用 --selftest)' % name)

    wv = WordVectors.load(a.model_prefix)
    print('讀入模型:%s(詞表 %d,維度 %d)' % (a.model_prefix, len(wv.words), wv.vectors.shape[1]))

    phrases_all = load_cpc_phrases(a.cpc_phrase_file)
    phrases, vecs, oov = embed_phrases(phrases_all, wv)
    print('CPC 短語總數 %d,可算出向量 %d,完全 OOV %d'
          % (len(phrases_all), len(phrases), len(oov)))
    if oov:
        print('OOV 範例(前 10 個): %s' % oov[:10])
    if len(phrases) < a.min_cluster_size:
        raise SystemExit('可用的 CPC 短語太少(%d 個),無法聚類,請檢查模型詞表或 CPC 檔案' % len(phrases))

    labels = hac_with_min_size(vecs, a.n_clusters, a.min_cluster_size, a.linkage)
    topics = build_topics(phrases, vecs, labels)
    save_topics(topics, a.out)

    sizes = sorted((len(v['phrases']) for v in topics.values()), reverse=True)
    print('\n得到 %d 個主題,各主題短語數: %s' % (len(topics), sizes))
    for i in sorted(topics)[:5]:
        print('  Topic %d (%d 個短語): %s' % (i, len(topics[i]['phrases']), topics[i]['phrases'][:8]))
    print('\n已存到 %s_centroids.npy / %s_phrases.json' % (a.out, a.out))


if __name__ == '__main__':
    main()