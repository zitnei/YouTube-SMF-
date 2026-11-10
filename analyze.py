"""音源解析: 曲がどんな方式の音源で鳴っているかを推定する。

  PSG系       … 矩形波・ノイズ（AY-3-8910, SN76489, ファミコンの矩形波 など）
  FM系        … 周波数変調（YM2612, YM2203/2608, OPL など）
  波形メモリ系 … 32サンプル程度の短い波形をくり返す（PCエンジン, SCC, ゲームボーイ波形ch など）
  PCM系       … 録音した音（サンプリング）を鳴らす（スーパーファミコン, GM/MIDI音源, 生演奏 など）

学習済みモデルは使わず、波形の特徴から点数をつける「推定」です。
"""

import numpy as np

SR = 44100
SEG = 4.0          # 4秒ごとに区切って解析
N_FFT = 4096
HOP = 1024
BIN = SR / N_FFT

TYPES = ["psg", "fm", "wavetable", "pcm"]
LABELS = {"psg": "PSG系", "fm": "FM系", "wavetable": "波形メモリ系", "pcm": "PCM系"}


# ---------- 特徴量 ----------

def _harmonics(spec: np.ndarray, f0: float, kmax: int) -> np.ndarray:
    """1〜kmax 倍音の振幅（各倍音の位置付近の最大値）。"""
    out = np.zeros(kmax + 1)
    n = len(spec)
    for k in range(1, kmax + 1):
        b = k * f0 / BIN
        lo, hi = int(np.floor(b - 1)), int(np.ceil(b + 1))
        if hi >= n:
            break
        out[k] = spec[max(lo, 0):hi + 1].max()
    return out


def _refine_f0(spec: np.ndarray, f0: float) -> float:
    """yin の推定を、倍音の山にぴったり合うように ±1.5% の範囲で微調整する。"""
    bins = np.arange(len(spec))
    best, best_s = f0, -1.0
    for f in f0 * np.linspace(0.985, 1.015, 31):
        ks = np.arange(1, 11) * f / BIN
        ks = ks[ks < len(spec) - 1]
        s = np.interp(ks, bins, spec).sum()
        if s > best_s:
            best, best_s = f, s
    return best


def segment_features(x: np.ndarray) -> dict | None:
    """4秒ぶんの音から特徴を測る。複数パートが同時に鳴っていても使える特徴だけを使う。"""
    import librosa

    if np.sqrt(np.mean(x ** 2)) < 3e-3:
        return None  # ほぼ無音
    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window="hann"))
    D = 20 * np.log10(S + 1e-9)
    energy = (S ** 2).sum(axis=0)
    active = energy > energy.max() * 10 ** (-3.0)  # 最大から30dB以内のフレーム
    if active.sum() < 8:
        return None
    top = int(min(0.45 * SR, 15500) / BIN)  # MP3 は 16kHz 以上が削られていることが多い
    lo_b, hi_b = int(100 / BIN), int(10000 / BIN)

    floors, flats, odds, imgs, holes = [], [], [], [], []
    cands = {}  # フレーム -> [(ビン, 倍音の相対dBベクトル)]

    def level(col, f):
        b = f / BIN
        lo, hi = int(np.floor(b - 1.5)), int(np.ceil(b + 1.5))
        if hi >= top:
            return None
        return col[max(lo, 0):hi + 1].max()

    for t in np.flatnonzero(active):
        col = D[:top, t]
        mx = col.max()
        # 1) 倍音と倍音のすき間の「底」の高さ（雑音・残響が多いと高い）
        # 圧縮音声（MP3/AAC/Opus）は小さな成分を丸ごと消して「穴」を作るので、穴は除いて測る
        band = col[lo_b:hi_b]
        band = band[band > mx - 100]
        if len(band) > 20:
            floors.append(np.percentile(band, 20) - mx)
        holes.append(1 - len(band) / max(hi_b - lo_b, 1))
        p = S[5:top, t] ** 2
        flats.append(np.exp(np.mean(np.log(p + 1e-12))) / (np.mean(p) + 1e-12))

        # 2) 強い山を1つずつ「基音」とみなして、倍音の並びを見る
        pk = np.flatnonzero((col[1:-1] > col[:-2]) & (col[1:-1] >= col[2:])) + 1
        pk = pk[(pk * BIN >= 55) & (pk * BIN <= 1000) & (col[pk] > mx - 12)]
        cands[t] = []
        for b in pk:
            # 放物線補間で正確な周波数に
            y0, y1, y2 = col[b - 1], col[b], col[b + 1]
            off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
            f = (b + off) * BIN
            a1 = col[b]
            r = {}
            for k in list(range(2, 9)) + list(range(20, 29)) + [31, 33]:
                lv = level(col, k * f)
                if lv is not None:
                    r[k] = lv - a1
            if 3 in r and 2 in r and r[3] > -25:
                odds.append(np.clip(r[3] - r[2], -20, 30))   # 矩形波: 3倍音はあるのに2倍音がない
            if 31 in r and 33 in r and all(k in r for k in range(20, 29)):
                imgs.append(max(r[31], r[33]) - max(r[k] for k in range(20, 29)))
            vec = [r.get(k, -80) for k in range(2, 9)]
            cands[t].append((b, np.maximum(vec, -60)))

    # 3) 同じ高さの音が続く間に、倍音のバランス（音色）がどれだけ変わるか
    drifts = []
    acts = np.flatnonzero(active)
    used = set()
    for t in acts:
        for b, v in cands.get(t, []):
            if (t, b) in used:
                continue
            run = [v]
            tt, bb = t, b
            while True:
                nxt = [(b2, v2) for b2, v2 in cands.get(tt + 1, []) if abs(b2 - bb) <= 1]
                if not nxt:
                    break
                bb, v2 = nxt[0]
                tt += 1
                used.add((tt, bb))
                run.append(v2)
            if len(run) >= 6:
                first = np.mean(run[:2], axis=0)
                last = np.mean(run[-2:], axis=0)
                strong = first > -40
                if strong.sum() >= 2:
                    drifts.append(float(np.mean(np.abs(first - last)[strong])))

    # 4) 波形の「段差」: 矩形波などは波形がほぼ平らで、ときどき大きく跳ぶ
    d = np.diff(x)
    kurt = float(np.mean(d ** 4) / (np.mean(d ** 2) ** 2 + 1e-12))

    # 5) 音量の「平らな区間」の割合
    db = 10 * np.log10(energy + 1e-12)
    dd = np.abs(np.diff(db))
    both = active[1:] & active[:-1]
    plateau = float(np.mean(dd[both] < 0.25)) if both.any() else np.nan

    med = lambda v: float(np.median(v)) if len(v) else np.nan
    return {
        "floor": med(floors),
        "flat": med(flats),
        "odd": med(odds),
        "odd_n": len(odds),
        "image": med(imgs),
        "drift": med(drifts),
        "runs": len(drifts),
        "kurt": kurt,
        "plateau": plateau,
        "holes": med(holes),
    }


# ---------- 点数づけ ----------

def _ramp(x, lo, hi):
    """lo→0, hi→1 の直線（範囲外は 0/1）。値がないときは中間の 0.5。"""
    if x is None or np.isnan(x):
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def indicators(f: dict) -> dict:
    """特徴量を 0〜1 の「〜らしさ」にする。"""
    lg = lambda v: np.log10(v) if v is not None and not np.isnan(v) and v > 0 else np.nan
    return {
        # 雑音・残響の多さ（PCM の録音・サンプリング音らしさ）
        "noisy": float(np.mean([_ramp(f["floor"], -60, -42), _ramp(lg(f["flat"]), -2.7, -1.3)])),
        # 波形の段差（矩形波＝PSG らしさ）
        "stepped": _ramp(lg(f["kurt"]), np.log10(13), np.log10(30)),
        # 31・33倍音の折り返し（波形メモリらしさ）
        "image": _ramp(f["image"], -3, 5),
        # 1音の中での音色変化（FM らしさ）
        "drift": _ramp(f["drift"], 7, 13),
    }


def score(f: dict) -> dict:
    i = indicators(f)
    clean = 1 - i["noisy"]
    chip = max(i["stepped"], i["image"], i["drift"])
    raw = {
        # チップ系は「その方式ならではの特徴」が見つかったときだけ点が上がる
        "psg":       clean * i["stepped"] * (1 - 0.3 * i["drift"]),
        "wavetable": clean * i["image"] * (1 - 0.3 * i["stepped"]),
        "fm":        clean * i["drift"] * (1 - 0.3 * i["stepped"]),
        # どのチップの特徴もない・雑音や残響が多い → PCM（録音・サンプリング音）
        "pcm":       0.6 * i["noisy"] + 0.4 * (1 - chip),
    }
    # 合計100%の割合にする（差を強調）
    e = {k: np.exp(5 * v) for k, v in raw.items()}
    s = sum(e.values())
    return {k: float(e[k] / s) for k in TYPES}


# ---------- 入口 ----------

FEATURE_KEYS = ["floor", "flat", "odd", "odd_n", "image", "drift", "runs", "kurt", "plateau", "holes"]


def analyze(wav_path: str) -> dict:
    import librosa

    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    if len(y) < SR:
        raise ValueError("音声が短すぎます（1秒以上必要です）")
    seg = int(SEG * SR)
    segments = []
    for start in range(0, len(y), seg):
        x = y[start:start + seg]
        if len(x) < seg // 2:
            break
        f = segment_features(x)
        if f is None:
            continue
        sc = score(f)
        segments.append({"start": start / SR, "end": (start + len(x)) / SR,
                         "type": max(sc, key=sc.get), "scores": sc, "features": f})
    if not segments:
        raise ValueError("音が小さすぎるか無音のため解析できませんでした")

    feats = {k: float(np.nanmedian([s["features"][k] for s in segments]))
             if not all(np.isnan(s["features"][k]) for s in segments) else np.nan
             for k in FEATURE_KEYS}
    # 全体の割合 = 区間ごとの割合の平均
    overall = {k: float(np.mean([s["scores"][k] for s in segments])) for k in TYPES}
    best = max(overall, key=overall.get)
    return {
        "duration": len(y) / SR,
        "best": best,
        "scores": overall,
        "family": "pcm" if best == "pcm" else "synth",
        "features": {k: (None if np.isnan(v) else round(v, 4)) for k, v in feats.items()},
        "evidence": evidence(feats),
        "timeline": [{"start": s["start"], "end": s["end"], "type": s["type"]} for s in segments],
    }


def evidence(f: dict) -> list[str]:
    """判定の根拠を日本語で。"""
    out = []
    has = lambda k: f.get(k) is not None and not np.isnan(f[k])
    if has("floor"):
        if f["floor"] >= -48:
            out.append("倍音と倍音のすき間に雑音や残響が多い（録音・サンプリングした音＝PCMの特徴）")
        elif f["floor"] <= -70:
            out.append("倍音以外の雑音がほとんどなく、とても電子的できれいな音")
    if has("kurt") and f["kurt"] >= 20:
        out.append("波形がほぼ平らで、ときどき大きく跳ぶ「段差」の形をしている（矩形波＝PSGの特徴）")
    if has("odd") and f["odd"] >= 15:
        out.append("偶数倍音が弱く奇数倍音が強い（矩形波・三角波の特徴）")
    if has("image") and f["image"] >= 2:
        out.append("31倍音・33倍音が不自然に強い（32サンプルの短い波形をくり返す波形メモリ音源の特徴）")
    if has("drift"):
        if f["drift"] >= 11:
            out.append("1つの音の中で音色（倍音のバランス）が大きく変わる（FM音源の典型的な特徴）")
        elif f["drift"] <= 6:
            out.append("1つの音の中で音色があまり変わらない")
    return out
