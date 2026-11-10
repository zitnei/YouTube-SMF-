// 音源解析（PSG / FM / 波形メモリ / PCM）のブラウザ版。
// 計算内容は ../../analyze.py と同じ。入力は 44.1kHz モノラルの Float32Array。

export const SR = 44100;
const SEG = 4.0;
const N_FFT = 4096;
const HOP = 1024;
const BIN = SR / N_FFT;
const NBINS = N_FFT / 2 + 1;

export const TYPES = ["psg", "fm", "wavetable", "pcm"];

// ---------- FFT（基数2） ----------

const WINDOW = new Float64Array(N_FFT).map((_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT));
const REV = (() => {
  const bits = Math.log2(N_FFT), r = new Uint32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    let x = i, y = 0;
    for (let b = 0; b < bits; b++) { y = (y << 1) | (x & 1); x >>= 1; }
    r[i] = y;
  }
  return r;
})();
const COS = new Float64Array(N_FFT / 2).map((_, i) => Math.cos((-2 * Math.PI * i) / N_FFT));
const SIN = new Float64Array(N_FFT / 2).map((_, i) => Math.sin((-2 * Math.PI * i) / N_FFT));

function fftMag(frame) {
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) re[REV[i]] = frame[i] * WINDOW[i];
  for (let size = 2; size <= N_FFT; size <<= 1) {
    const half = size >> 1, step = N_FFT / size;
    for (let i = 0; i < N_FFT; i += size) {
      for (let j = 0; j < half; j++) {
        const k = j * step, a = i + j, b = a + half;
        const tr = re[b] * COS[k] - im[b] * SIN[k];
        const ti = re[b] * SIN[k] + im[b] * COS[k];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
  const mag = new Float64Array(NBINS);
  for (let i = 0; i < NBINS; i++) mag[i] = Math.hypot(re[i], im[i]);
  return mag;
}

// librosa.stft(center=True) と同じく、前後を 0 で埋めてから切り出す
function stft(x) {
  const pad = N_FFT / 2, n = 1 + Math.floor(x.length / HOP), frames = [];
  const buf = new Float64Array(N_FFT);
  for (let t = 0; t < n; t++) {
    const start = t * HOP - pad;
    for (let i = 0; i < N_FFT; i++) {
      const j = start + i;
      buf[i] = j >= 0 && j < x.length ? x[j] : 0;
    }
    frames.push(fftMag(buf));
  }
  return frames;
}

// ---------- 小さな道具 ----------

const median = (v) => {
  if (!v.length) return NaN;
  const s = [...v].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const percentile = (v, q) => {
  const s = [...v].sort((a, b) => a - b), pos = (q / 100) * (s.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
};
const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
const isNum = (v) => typeof v === "number" && !Number.isNaN(v);

// ---------- 特徴量（analyze.py の segment_features と同じ） ----------

export function segmentFeatures(x) {
  let sq = 0;
  for (const v of x) sq += v * v;
  if (Math.sqrt(sq / x.length) < 3e-3) return null;

  const S = stft(x);
  const T = S.length;
  const energy = S.map((col) => col.reduce((a, v) => a + v * v, 0));
  const emax = Math.max(...energy);
  const active = energy.map((e) => e > emax * 1e-3);
  if (active.filter(Boolean).length < 8) return null;

  const top = Math.floor(Math.min(0.45 * SR, 15500) / BIN);
  const loB = Math.floor(100 / BIN), hiB = Math.floor(10000 / BIN);
  const floors = [], flats = [], odds = [], imgs = [], holes = [];
  const cands = new Map();
  const KS = [2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 33];

  for (let t = 0; t < T; t++) {
    if (!active[t]) continue;
    const col = new Float64Array(top);
    let mx = -Infinity;
    for (let i = 0; i < top; i++) { col[i] = 20 * Math.log10(S[t][i] + 1e-9); if (col[i] > mx) mx = col[i]; }

    // 1) 倍音のすき間の底（圧縮でできた「穴」は除く）
    const band = [];
    for (let i = loB; i < hiB; i++) if (col[i] > mx - 100) band.push(col[i]);
    if (band.length > 20) floors.push(percentile(band, 20) - mx);
    holes.push(1 - band.length / Math.max(hiB - loB, 1));

    // スペクトルの平たさ
    let lsum = 0, psum = 0;
    for (let i = 5; i < top; i++) { const p = S[t][i] ** 2; lsum += Math.log(p + 1e-12); psum += p; }
    const cnt = top - 5;
    flats.push(Math.exp(lsum / cnt) / (psum / cnt + 1e-12));

    // 2) 強い山を基音とみなして倍音の並びを見る
    const level = (f) => {
      const b = f / BIN, lo = Math.floor(b - 1.5), hi = Math.ceil(b + 1.5);
      if (hi >= top) return null;
      let m = -Infinity;
      for (let i = Math.max(lo, 0); i <= hi; i++) if (col[i] > m) m = col[i];
      return m;
    };
    const list = [];
    for (let b = 1; b < top - 1; b++) {
      if (!(col[b] > col[b - 1] && col[b] >= col[b + 1])) continue;
      if (b * BIN < 55 || b * BIN > 1000 || col[b] <= mx - 12) continue;
      const y0 = col[b - 1], y1 = col[b], y2 = col[b + 1];
      const off = (0.5 * (y0 - y2)) / (y0 - 2 * y1 + y2 + 1e-12);
      const f = (b + off) * BIN, a1 = col[b];
      const r = {};
      for (const k of KS) { const lv = level(k * f); if (lv !== null) r[k] = lv - a1; }
      if (3 in r && 2 in r && r[3] > -25) odds.push(Math.min(30, Math.max(-20, r[3] - r[2])));
      let all = 31 in r && 33 in r;
      for (let k = 20; k <= 28 && all; k++) all = k in r;
      if (all) {
        let m = -Infinity;
        for (let k = 20; k <= 28; k++) m = Math.max(m, r[k]);
        imgs.push(Math.max(r[31], r[33]) - m);
      }
      const vec = [];
      for (let k = 2; k <= 8; k++) vec.push(Math.max(k in r ? r[k] : -80, -60));
      list.push([b, vec]);
    }
    cands.set(t, list);
  }

  // 3) 同じ高さの音が続く間の音色（倍音バランス）の変化
  const drifts = [], used = new Set();
  for (let t = 0; t < T; t++) {
    if (!active[t]) continue;
    for (const [b, v] of cands.get(t) || []) {
      if (used.has(t * 4096 + b)) continue;
      const run = [v];
      let tt = t, bb = b;
      for (;;) {
        const nxt = (cands.get(tt + 1) || []).filter(([b2]) => Math.abs(b2 - bb) <= 1);
        if (!nxt.length) break;
        bb = nxt[0][0]; tt += 1;
        used.add(tt * 4096 + bb);
        run.push(nxt[0][1]);
      }
      if (run.length >= 6) {
        const first = run[0].map((_, i) => (run[0][i] + run[1][i]) / 2);
        const n = run.length;
        const last = run[0].map((_, i) => (run[n - 2][i] + run[n - 1][i]) / 2);
        const diffs = [];
        first.forEach((fv, i) => { if (fv > -40) diffs.push(Math.abs(fv - last[i])); });
        if (diffs.length >= 2) drifts.push(mean(diffs));
      }
    }
  }

  // 4) 波形の段差（差分の尖度）
  let m2 = 0, m4 = 0;
  for (let i = 1; i < x.length; i++) { const d = x[i] - x[i - 1], d2 = d * d; m2 += d2; m4 += d2 * d2; }
  const nd = x.length - 1;
  const kurt = (m4 / nd) / ((m2 / nd) ** 2 + 1e-12);

  // 5) 音量の平らな区間
  const db = energy.map((e) => 10 * Math.log10(e + 1e-12));
  const flatsteps = [];
  for (let t = 1; t < T; t++) if (active[t] && active[t - 1]) flatsteps.push(Math.abs(db[t] - db[t - 1]) < 0.25 ? 1 : 0);

  return {
    floor: median(floors), flat: median(flats), odd: median(odds), odd_n: odds.length,
    image: median(imgs), drift: median(drifts), runs: drifts.length, kurt,
    plateau: flatsteps.length ? mean(flatsteps) : NaN, holes: median(holes),
  };
}

// ---------- 点数づけ ----------

const ramp = (x, lo, hi) => (isNum(x) ? Math.min(1, Math.max(0, (x - lo) / (hi - lo))) : 0.5);
const lg = (v) => (isNum(v) && v > 0 ? Math.log10(v) : NaN);

export function indicators(f) {
  return {
    noisy: (ramp(f.floor, -60, -42) + ramp(lg(f.flat), -2.7, -1.3)) / 2,
    stepped: ramp(lg(f.kurt), Math.log10(13), Math.log10(30)),
    image: ramp(f.image, -3, 5),
    drift: ramp(f.drift, 7, 13),
  };
}

export function score(f) {
  const i = indicators(f);
  const clean = 1 - i.noisy;
  const chip = Math.max(i.stepped, i.image, i.drift);
  const raw = {
    psg: clean * i.stepped * (1 - 0.3 * i.drift),
    wavetable: clean * i.image * (1 - 0.3 * i.stepped),
    fm: clean * i.drift * (1 - 0.3 * i.stepped),
    pcm: 0.6 * i.noisy + 0.4 * (1 - chip),
  };
  const e = Object.fromEntries(TYPES.map((k) => [k, Math.exp(5 * raw[k])]));
  const s = TYPES.reduce((a, k) => a + e[k], 0);
  return Object.fromEntries(TYPES.map((k) => [k, e[k] / s]));
}

// ---------- 入口 ----------

const KEYS = ["floor", "flat", "odd", "odd_n", "image", "drift", "runs", "kurt", "plateau", "holes"];

export async function analyze(y, onProgress = () => {}) {
  if (y.length < SR) throw new Error("音声が短すぎます（1秒以上必要です）");
  const seg = Math.round(SEG * SR);
  const segments = [];
  for (let start = 0; start < y.length; start += seg) {
    const x = y.subarray(start, start + seg);
    if (x.length < seg / 2) break;
    const f = segmentFeatures(x);
    onProgress(Math.min(1, (start + seg) / y.length));
    await new Promise((r) => setTimeout(r)); // 画面が固まらないように
    if (!f) continue;
    const sc = score(f);
    const type = TYPES.reduce((a, k) => (sc[k] > sc[a] ? k : a), TYPES[0]);
    segments.push({ start: start / SR, end: (start + x.length) / SR, type, scores: sc, features: f });
  }
  if (!segments.length) throw new Error("音が小さすぎるか無音のため解析できませんでした");

  const feats = {};
  for (const k of KEYS) feats[k] = median(segments.map((s) => s.features[k]).filter(isNum));
  const overall = Object.fromEntries(TYPES.map((k) => [k, mean(segments.map((s) => s.scores[k]))]));
  const best = TYPES.reduce((a, k) => (overall[k] > overall[a] ? k : a), TYPES[0]);
  return {
    duration: y.length / SR, best, scores: overall, features: feats,
    evidence: evidence(feats),
    timeline: segments.map((s) => ({ start: s.start, end: s.end, type: s.type })),
  };
}

function evidence(f) {
  const out = [];
  if (isNum(f.floor)) {
    if (f.floor >= -48) out.push("倍音と倍音のすき間に雑音や残響が多い（録音・サンプリングした音＝PCMの特徴）");
    else if (f.floor <= -70) out.push("倍音以外の雑音がほとんどなく、とても電子的できれいな音");
  }
  if (isNum(f.kurt) && f.kurt >= 20) out.push("波形がほぼ平らで、ときどき大きく跳ぶ「段差」の形をしている（矩形波＝PSGの特徴）");
  if (isNum(f.odd) && f.odd >= 15) out.push("偶数倍音が弱く奇数倍音が強い（矩形波・三角波の特徴）");
  if (isNum(f.image) && f.image >= 2) out.push("31倍音・33倍音が不自然に強い（32サンプルの短い波形をくり返す波形メモリ音源の特徴）");
  if (isNum(f.drift)) {
    if (f.drift >= 11) out.push("1つの音の中で音色（倍音のバランス）が大きく変わる（FM音源の典型的な特徴）");
    else if (f.drift <= 6) out.push("1つの音の中で音色があまり変わらない");
  }
  return out;
}
