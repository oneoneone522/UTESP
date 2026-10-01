# -*- coding: utf-8 -*-
"""
check_phrase_vectors.py -- 驗證已訓練好的詞向量,對「短語」而不是單詞夠不夠用。

檢查兩件事:
  1. 覆蓋率:candidate_merge.py 產生的 _superspan_sequence.json 裡,
     有多少候選短語能算出向量(短語裡至少有一個詞在詞表內)。
  2. 品質:抽幾個能算出向量的短語,看它們的最近鄰短語像不像同類技術詞彙。

用法:
    python3 check_phrase_vectors.py models/sg_small \
        techpat-data/Electricity/example_data/example_title/title.txt_superspan_sequence.json
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, '.')
from skipgram import WordVectors


def load_superspan_phrases(path):
    """從 candidate_merge.py 的輸出裡,把每篇文件的 superspan 短語文字攤平成一個清單。"""
    phrases = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            seq = json.loads(line)
            for span in seq:
                if span.get('tag') == 'superspan':
                    phrases.append(span['text'])
    return phrases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_prefix', help='例如 models/sg_small(不含 .npy)')
    ap.add_argument('superspan_json', nargs='?', default=None,
                     help='candidate_merge.py 的輸出檔,例如 xxx_superspan_sequence.json')
    ap.add_argument('--topn', type=int, default=5)
    a = ap.parse_args()

    wv = WordVectors.load(a.model_prefix)
    print('讀入模型:%s(詞表 %d,維度 %d)\n' % (a.model_prefix, len(wv.words), wv.vectors.shape[1]))

    # --- 1. 手動挑幾個候選短語,直接看向量算不算得出來、鄰居合不合理 ---
    demo_phrases = ['printed circuit', 'copper foil', 'semiconductor wafers',
                     'antenna structure', 'wireless communication device',
                     'conductive semiconductor material']
    print('== 手動挑的短語 ==')
    phrase_vecs = {}
    for p in demo_phrases:
        v = wv.phrase_vector(p)
        oov = [w for w in p.lower().split() if w not in wv]
        if v is None:
            print('%-32s -> 完全 OOV,無法算向量' % p)
        else:
            phrase_vecs[p] = v
            tag = '' if not oov else '(缺詞: %s)' % ','.join(oov)
            print('%-32s -> 向量前3維 %s %s' % (p, np.round(v[:3], 3), tag))
    print()

    if len(phrase_vecs) >= 2:
        print('== 短語之間的 cosine 相似度(檢查語意是否合理相近/相遠) ==')
        items = list(phrase_vecs.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (p1, v1), (p2, v2) = items[i], items[j]
                sim = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
                print('  %-30s vs %-30s : %.3f' % (p1, p2, sim))
        print()

    # --- 2. 如果有給真正的候選短語檔,統計整體覆蓋率 ---
    if a.superspan_json:
        phrases = load_superspan_phrases(a.superspan_json)
        n = len(phrases)
        n_ok = sum(1 for p in phrases if wv.phrase_vector(p) is not None)
        n_full_oov = n - n_ok
        print('== 候選短語覆蓋率(%s) ==' % a.superspan_json)
        print('候選短語總數: %d' % n)
        print('可算出向量:   %d (%.1f%%)' % (n_ok, 100 * n_ok / n if n else 0))
        print('完全 OOV:     %d (%.1f%%)  <- 這些短語目前無法參與建圖/評分' % (
            n_full_oov, 100 * n_full_oov / n if n else 0))

        oov_examples = [p for p in phrases if wv.phrase_vector(p) is None]
        if oov_examples:
            print('完全 OOV 的短語範例(前 10 個): %s' % oov_examples[:10])


if __name__ == '__main__':
    main()