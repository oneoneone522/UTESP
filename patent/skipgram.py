# -*- coding: utf-8 -*-
"""
skipgram.py -- 依 GeeksforGeeks〈Implement your own word2vec(skip-gram) model in Python〉的數學實作 skip-gram,
並改成可以讀專利語料(一行一份文件、詞以空白分隔、已小寫)。

網站的公式(一次處理一組 center -> context):
    h = W^T x              x 是 one-hot,等於取 W 的第 center 列
    u = W'^T h
    y = softmax(u)
    E = -log y[context]
    dE/du = y - t ,  dE/dW' = h (y - t)^T ,  dE/dW = x (W'(y - t))^T
numpy 後端照這組公式;為了跑得動,改成用「詞編號」取代 one-hot 向量、用 mini-batch 訓練。

用法:
    python3 skipgram.py --selftest                                   # 先確認公式與程式沒寫錯
    python3 skipgram.py --inputs title.txt abstract.txt --out models/sg_small --max_lines 2000
    python3 skipgram.py --backend gensim --inputs ... --out models/sg_full   # 全量語料建議用這個

之後在別的程式裡:
    from skipgram import WordVectors
    wv = WordVectors.load('models/sg_full')
    wv.phrase_vector('printed circuit')      # 短語向量 = 詞向量平均
"""
import argparse
import json
import os
import re
import time
from array import array
from collections import Counter

import numpy as np

_HAS_ALNUM = re.compile(r'[A-Za-z0-9]')


# ----------------------------------------------------------------------------
# 語料
# ----------------------------------------------------------------------------
class Corpus:
    """可重複掃描的語料(建詞表要掃一次、編碼要掃一次,gensim 也會多次掃描)。

    每個檔案:一行一份文件,詞以空白分隔。以獨立的 '.' token 斷句,丟掉純標點 token。
    不移除 stopword:候選短語裡會出現 of / for 之類的詞,之後算短語向量要用到。
    """

    def __init__(self, paths, max_lines=None):
        self.paths = paths
        self.max_lines = max_lines

    def __iter__(self):
        for path in self.paths:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                for n, line in enumerate(f):
                    if self.max_lines is not None and n >= self.max_lines:
                        break
                    sent = []
                    for tok in line.split():
                        if tok == '.':
                            if len(sent) > 1:
                                yield sent
                            sent = []
                        elif _HAS_ALNUM.search(tok):
                            sent.append(tok)
                    if len(sent) > 1:
                        yield sent


def build_vocab(sentences, min_count=5, max_vocab=None):
    counter = Counter()
    for s in sentences:
        counter.update(s)
    items = sorted(((w, c) for w, c in counter.items() if c >= min_count),
                   key=lambda x: (-x[1], x[0]))
    if max_vocab:
        items = items[:max_vocab]
    words = [w for w, _ in items]
    return words, {w: i for i, w in enumerate(words)}


def encode(sentences, vocab):
    """攤平成兩條陣列:ids(詞編號)、sid(所屬句子編號)。不在詞表內的詞直接丟掉(同 word2vec)。"""
    ids, sid = array('i'), array('i')
    k = 0
    for s in sentences:
        row = [vocab[w] for w in s if w in vocab]
        if len(row) > 1:
            ids.extend(row)
            sid.extend([k] * len(row))
            k += 1
    if not ids:
        return np.zeros(0, dtype=np.intc), np.zeros(0, dtype=np.intc)
    return np.frombuffer(ids, dtype=np.intc), np.frombuffer(sid, dtype=np.intc)


def count_pairs(sid, window):
    total = 0
    for off in range(1, window + 1):
        if len(sid) > off:
            total += int((sid[:-off] == sid[off:]).sum())
    return 2 * total  # (中心詞->脈絡詞) 兩個方向


def iter_batches(ids, sid, window, batch_size, rng, chunk=200_000):
    """依需要產生 (center, context) 訓練對,一塊塊處理,不必一次把所有配對放進記憶體。"""
    n = len(ids)
    starts = np.arange(0, n, chunk)
    rng.shuffle(starts)
    for start in starts:
        end = min(n, start + chunk)
        cen, ctx = [], []
        for off in range(1, window + 1):
            hi = min(end + off, n)
            if hi - off <= start:
                continue
            a = ids[start:hi - off]
            b = ids[start + off:hi]
            same = sid[start:hi - off] == sid[start + off:hi]   # 不跨句子
            a, b = a[same], b[same]
            cen += [a, b]
            ctx += [b, a]
        if not cen:
            continue
        cen = np.concatenate(cen)
        ctx = np.concatenate(ctx)
        perm = rng.permutation(len(cen))
        cen, ctx = cen[perm], ctx[perm]
        for i in range(0, len(cen), batch_size):
            yield cen[i:i + batch_size], ctx[i:i + batch_size]


# ----------------------------------------------------------------------------
# 模型(對應網站的 word2vec class)
# ----------------------------------------------------------------------------
class SkipGram:
    def __init__(self, vocab_size, dim=100, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.V, self.N = vocab_size, dim
        # W : 輸入層 -> 隱藏層 (V x N),訓練完的每一列就是該詞的向量
        self.W = ((rng.random((self.V, self.N)) - 0.5) / self.N).astype(dtype)
        # W1: 隱藏層 -> 輸出層 (N x V),對應網站的 W'
        self.W1 = np.zeros((self.N, self.V), dtype=dtype)

    def loss_and_grads(self, centers, contexts):
        """一個 batch 的『總』loss 與精確梯度(= 把網站的單筆公式對 batch 內每一筆加總)。"""
        B = len(centers)
        h = self.W[centers]                       # (B,N)  h = W^T x
        u = h @ self.W1                           # (B,V)  u = W'^T h
        u -= u.max(axis=1, keepdims=True)
        y = np.exp(u)
        y /= y.sum(axis=1, keepdims=True)         # softmax
        rows = np.arange(B)
        loss = float(-np.log(y[rows, contexts] + 1e-12).sum())
        e = y                                     # e = y - t   (t 是 context 的 one-hot)
        e[rows, contexts] -= 1.0
        dW1 = h.T @ e                             # (N,V)  dE/dW' = h e^T
        dh = e @ self.W1.T                        # (B,N)  W' e;第 i 列要加到 W[centers[i]]
        return loss, dW1, dh

    def fit(self, ids, sid, window=5, epochs=3, lr=0.025, batch_size=64, seed=0, log=print):
        rng = np.random.default_rng(seed)
        total_pairs = count_pairs(sid, window)
        total_batches = max(1, epochs * -(-total_pairs // batch_size))
        step = 0
        for ep in range(1, epochs + 1):
            t0, run, seen = time.time(), 0.0, 0
            for cen, ctx in iter_batches(ids, sid, window, batch_size, rng):
                cur = max(lr * 0.05, lr * (1 - step / total_batches))   # 線性遞減的學習率
                loss, dW1, dh = self.loss_and_grads(cen, ctx)
                self.W1 -= cur * dW1
                np.add.at(self.W, cen, -cur * dh)     # 同一個詞在 batch 內出現多次要累加
                run += loss
                seen += len(cen)
                step += 1
                if step % 500 == 0:
                    log('epoch %d/%d  batch %d/%d  平均 loss %.4f  lr %.4f'
                        % (ep, epochs, step, total_batches, run / seen, cur))
            log('== epoch %d 完成:平均 loss %.4f,耗時 %.0f 秒' % (ep, run / max(seen, 1), time.time() - t0))

    def predict(self, center_id, topn=3):
        """對應網站的 predict:給一個中心詞,回傳機率最高的 topn 個脈絡詞的編號。"""
        u = self.W[center_id][None, :] @ self.W1
        return np.argsort(-u[0])[:topn]     # softmax 是單調的,直接排 u 即可


# ----------------------------------------------------------------------------
# 詞向量(不論哪個後端訓練,下游只用這個類別)
# ----------------------------------------------------------------------------
class WordVectors:
    def __init__(self, words, vectors):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def __contains__(self, w):
        return w in self.index

    def vector(self, w):
        return self.vectors[self.index[w]]

    def phrase_vector(self, phrase):
        """短語向量 = 短語中『詞表內的詞』的向量平均;整個短語都不在詞表內時回傳 None。"""
        vecs = [self.vectors[self.index[w]] for w in phrase.lower().split() if w in self.index]
        return np.mean(vecs, axis=0) if vecs else None

    def most_similar(self, w, topn=5):
        v = self.vector(w)
        sims = self.vectors @ v / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(v) + 1e-12)
        order = np.argsort(-sims)
        return [(self.words[i], float(sims[i])) for i in order if self.words[i] != w][:topn]

    def save(self, prefix):
        os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)
        np.save(prefix + '.npy', self.vectors)
        with open(prefix + '.vocab.json', 'w', encoding='utf-8') as f:
            json.dump(self.words, f, ensure_ascii=False)

    @classmethod
    def load(cls, prefix):
        with open(prefix + '.vocab.json', encoding='utf-8') as f:
            words = json.load(f)
        return cls(words, np.load(prefix + '.npy'))


# ----------------------------------------------------------------------------
# 訓練入口
# ----------------------------------------------------------------------------
def train_numpy(corpus, a):
    words, vocab = build_vocab(corpus, a.min_count, a.max_vocab)
    ids, sid = encode(corpus, vocab)
    if len(ids) == 0:
        raise SystemExit('沒有可訓練的詞,請檢查輸入檔與 --min_count')
    pairs = count_pairs(sid, a.window)
    print('詞表大小 V=%d,token 數 %d,每個 epoch 約 %d 組 (center, context)' % (len(words), len(ids), pairs))
    print('每個 batch 要算 %d x %d 的 softmax(約 %.0f MB);V 越大越慢,這是完整 softmax 的代價'
          % (a.batch, len(words), a.batch * len(words) * 4 / 1e6))
    model = SkipGram(len(words), a.dim, a.seed)
    model.fit(ids, sid, a.window, a.epochs, a.lr, a.batch, a.seed)
    return WordVectors(words, model.W)


def train_gensim(corpus, a):
    from gensim.models import Word2Vec
    model = Word2Vec(corpus, vector_size=a.dim, window=a.window, min_count=a.min_count, sg=1,
                     negative=5, epochs=a.epochs, workers=a.workers, seed=a.seed,
                     max_final_vocab=a.max_vocab)
    return WordVectors(model.wv.index_to_key, model.wv.vectors)


def selftest():
    rng = np.random.default_rng(0)
    # 1) 梯度檢查:解析梯度 vs 數值微分
    m = SkipGram(6, 4, seed=1, dtype=np.float64)
    m.W = rng.normal(size=(6, 4)) * 0.5
    m.W1 = rng.normal(size=(4, 6)) * 0.5
    c = np.array([0, 1, 1, 3])
    t = np.array([2, 2, 4, 5])
    _, dW1, dh = m.loss_and_grads(c, t)
    dW = np.zeros_like(m.W)
    np.add.at(dW, c, dh)
    eps, worst = 1e-6, 0.0
    for P, G in ((m.W1, dW1), (m.W, dW)):
        for idx in np.ndindex(*P.shape):
            old = P[idx]
            P[idx] = old + eps
            lp = m.loss_and_grads(c, t)[0]
            P[idx] = old - eps
            lm = m.loss_and_grads(c, t)[0]
            P[idx] = old
            worst = max(worst, abs((lp - lm) / (2 * eps) - G[idx]))
    print('梯度檢查:最大誤差 %.2e  %s' % (worst, '通過' if worst < 1e-6 else '失敗'))

    # 2) 網站的玩具語料
    corpus = [s.split() for s in ('earth revolves around sun', 'moon revolves around earth')]
    words, vocab = build_vocab(corpus, min_count=1)
    ids, sid = encode(corpus, vocab)
    model = SkipGram(len(words), 10, seed=0)
    model.fit(ids, sid, window=2, epochs=400, lr=0.2, batch_size=8, log=lambda *_: None)
    top = [words[i] for i in model.predict(vocab['around'], 3)]
    ok = {'revolves', 'earth'} <= set(top)
    print("'around' 最可能的脈絡詞:%s  %s" % (top, '通過' if ok else '失敗(預期含 revolves 與 earth)'))


def main():
    ap = argparse.ArgumentParser(description='skip-gram 詞向量訓練')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--inputs', nargs='+', help='語料檔(一行一份文件、空白分詞)')
    ap.add_argument('--out', help='輸出前綴,例如 models/sg_full')
    ap.add_argument('--backend', choices=['numpy', 'gensim'], default='numpy')
    ap.add_argument('--dim', type=int, default=100)
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=0.025)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--min_count', type=int, default=5)
    ap.add_argument('--max_vocab', type=int, default=None)
    ap.add_argument('--max_lines', type=int, default=None, help='每個檔案只讀前 N 行(先小規模試跑用)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4, help='gensim 後端的執行緒數')
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.inputs or not a.out:
        ap.error('需要 --inputs 與 --out(或用 --selftest)')

    corpus = Corpus(a.inputs, a.max_lines)
    t0 = time.time()
    wv = train_numpy(corpus, a) if a.backend == 'numpy' else train_gensim(corpus, a)
    wv.save(a.out)
    print('已存到 %s.npy / %s.vocab.json(詞表 %d,維度 %d,耗時 %.0f 秒)'
          % (a.out, a.out, len(wv.words), wv.vectors.shape[1], time.time() - t0))
    for w in ('circuit', 'semiconductor', 'antenna', 'wafer'):
        if w in wv:
            print(w, '->', [x for x, _ in wv.most_similar(w, 5)])


if __name__ == '__main__':
    main()