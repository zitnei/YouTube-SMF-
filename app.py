"""音楽 → MIDI 変換サイト

MP3 / WAV などの音声ファイル、または YouTube のリンクから
標準MIDIファイル (SMF, .mid) を作ります。
「パート分け」をオンにすると、AI で ボーカル / ベース / ドラム / その他 に
分離してから、パートごとに別トラックの MIDI にします。

起動:  .venv\\Scripts\\python app.py   →  http://127.0.0.1:8000
"""

import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import pretty_midi
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict

BASE_DIR = Path(__file__).parent
MAX_UPLOAD_MB = 100
MAX_YOUTUBE_SECONDS = 15 * 60  # 長すぎる動画は時間がかかるので制限

app = FastAPI(title="音楽 → MIDI 変換")
model = Model(ICASSP_2022_MODEL_PATH)  # 起動時に1回だけ読み込む

# パートごとの設定: MIDI 音色番号（GM, 0始まり）と、音程を探す周波数の範囲
PARTS = {
    "vocals": {"name": "Vocals", "program": 53, "min_freq": 120, "max_freq": 1400},  # Voice Oohs（低すぎる音はベースのにじみ）
    "bass":   {"name": "Bass",   "program": 33, "min_freq": 30, "max_freq": 400},    # Electric Bass
    "other":  {"name": "Other",  "program": 0,  "min_freq": None, "max_freq": None}, # Piano
    "drums":  {"name": "Drums",  "program": 0,  "drum": True},
}
PART_ORDER = ["vocals", "other", "bass", "drums"]


# ---------- 共通処理 ----------

class Settings(BaseModel):
    onset: float = 0.5        # 音の出だしの検出しきい値（低いほど音が増える）
    frame: float = 0.3        # 音が鳴り続けている判定のしきい値
    min_note_ms: float = 120  # これより短い音は捨てる
    min_freq: float | None = None
    max_freq: float | None = None
    tempo: float = 120
    separate: bool = False    # パート分けするか
    parts: list[str] = PART_ORDER
    smf_format: int = 1       # SMF フォーマット 0 か 1


def ffmpeg(src: Path, dst: Path, rate: int, channels: int) -> None:
    """どんな音声形式でも ffmpeg で WAV にそろえる。"""
    if not shutil.which("ffmpeg"):
        raise HTTPException(500, "ffmpeg が見つかりません。インストールしてください。")
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", str(channels), "-ar", str(rate), str(dst)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise HTTPException(400, "音声ファイルを読み込めませんでした。形式を確認してください。")


def transcribe(wav: Path, s: Settings, min_freq=None, max_freq=None) -> pretty_midi.Instrument:
    """音程のある音を basic-pitch でノートにする。"""
    _, midi, _ = predict(
        str(wav),
        model,
        onset_threshold=s.onset,
        frame_threshold=s.frame,
        minimum_note_length=s.min_note_ms,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        midi_tempo=s.tempo,
    )
    inst = pretty_midi.Instrument(program=0)
    for i in midi.instruments:
        inst.notes.extend(i.notes)
        inst.pitch_bends.extend(i.pitch_bends)
    inst.notes.sort(key=lambda n: n.start)
    return inst


# ---------- パート分け（Demucs） ----------

_demucs = None
_demucs_lock = threading.Lock()  # 分離はメモリを多く使うので同時に1つだけ


def separate(src: Path, workdir: Path) -> dict[str, Path]:
    """曲を drums / bass / other / vocals の4つの音声に分ける。"""
    global _demucs
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    wav44 = workdir / "stereo44.wav"
    ffmpeg(src, wav44, 44100, 2)

    with _demucs_lock:
        if _demucs is None:
            _demucs = get_model("htdemucs")
            _demucs.eval()
        m = _demucs
        device = "cuda" if torch.cuda.is_available() else "cpu"

        data, _ = sf.read(str(wav44), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T.copy())
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            out = apply_model(m, ((wav - mean) / std)[None], device=device,
                              split=True, overlap=0.25, progress=False)[0]
        out = out * std + mean

    stems = {}
    for name, audio in zip(m.sources, out):
        p = workdir / f"stem_{name}.wav"
        sf.write(str(p), audio.cpu().numpy().T, m.samplerate)
        stems[name] = p
    return stems


def transcribe_drums(wav: Path) -> pretty_midi.Instrument:
    """ドラムは音程がないので、帯域ごとの音の立ち上がりでキック/スネア/ハイハットを打ち込む。"""
    import librosa

    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    hop = 256
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    if np.max(np.abs(y)) < 1e-3:
        return inst

    bands = [  # (MIDI ドラム番号, 低い周波数, 高い周波数, 感度)
        (36, 30, 150, 0.12),     # キック
        (38, 400, 4000, 0.15),   # スネア
        (42, 6000, 11000, 0.12), # ハイハット
    ]
    # 帯域ごとの「音の立ち上がりの強さ」
    fluxes = []
    for pitch, lo, hi, delta in bands:
        band = spec[(freqs >= lo) & (freqs < hi)]
        flux = np.maximum(0, np.diff(np.log1p(band * 10), axis=1, prepend=0)).mean(axis=0) if band.size else np.zeros(1)
        fluxes.append(flux)
    loudest = max(float(np.percentile(f, 99.5)) for f in fluxes)
    if loudest <= 0:
        return inst

    hits = {}
    envs = {}
    for (pitch, lo, hi, delta), flux in zip(bands, fluxes):
        # 他の帯域よりずっと弱い帯域は「にじみ」とみなして小さく扱う
        scale = max(float(np.percentile(flux, 99.5)), loudest * 0.35)
        env = np.clip(flux / scale, 0, 1)
        frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop,
                                            delta=delta, wait=4, backtrack=False, normalize=False)
        hits[pitch] = [f for f in frames if env[f] >= 0.3]
        envs[pitch] = env

    # キックやハイハットのアタック音は中音域にも漏れるので、
    # 同じ瞬間のスネアは「はっきり強いとき」だけ残す
    others = np.array(sorted(hits[36] + hits[42]))
    def near_other(f):
        return others.size and np.min(np.abs(others - f)) <= 2
    hits[38] = [f for f in hits[38] if not near_other(f) or envs[38][f] >= 0.75]

    for pitch, frames in hits.items():
        env = envs[pitch]
        for f in frames:
            t = float(librosa.frames_to_time(f, sr=sr, hop_length=hop))
            vel = int(np.clip(50 + 77 * env[f], 1, 127))
            inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=t, end=t + 0.1))
    inst.notes.sort(key=lambda n: n.start)
    return inst


# ---------- 変換の入口 ----------

def convert(src: Path, workdir: Path, s: Settings) -> tuple[pretty_midi.PrettyMIDI, dict[str, int]]:
    pm = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=s.tempo)
    counts: dict[str, int] = {}

    if not s.separate:
        wav = workdir / "audio.wav"
        ffmpeg(src, wav, 22050, 1)
        inst = transcribe(wav, s, s.min_freq, s.max_freq)
        inst.name = "Music"
        pm.instruments.append(inst)
        counts["all"] = len(inst.notes)
        return pm, counts

    parts = [p for p in PART_ORDER if p in s.parts]
    if not parts:
        raise HTTPException(400, "パートを1つ以上選んでください。")
    stems = separate(src, workdir)
    for part in parts:
        cfg = PARTS[part]
        if cfg.get("drum"):
            inst = transcribe_drums(stems[part])
        else:
            inst = transcribe(stems[part], s, cfg["min_freq"], cfg["max_freq"])
            inst.program = cfg["program"]
        inst.name = cfg["name"]
        counts[part] = len(inst.notes)
        if inst.notes:
            pm.instruments.append(inst)
    return pm, counts


def write_smf(pm: pretty_midi.PrettyMIDI, out: Path, smf_format: int) -> None:
    """標準MIDIファイル (SMF) として書き出す。
    フォーマット1: パートごとに別トラック（DAW 向け）
    フォーマット0: 全部を1トラックにまとめる（古い機器・カラオケ・ゲーム機など向け）
    どちらもパートの区別は MIDI チャンネルで保たれる（ドラムは10ch）。"""
    import mido

    pm.write(str(out))
    if smf_format != 0:
        return
    src = mido.MidiFile(str(out))
    merged = mido.merge_tracks(src.tracks)
    # 各パートのトラック名は1トラックにまとめると意味がないので外す
    track = mido.MidiTrack()
    pending = 0
    for msg in merged:
        if msg.is_meta and msg.type in ("track_name", "end_of_track"):
            pending += msg.time
            continue
        track.append(msg.copy(time=msg.time + pending))
        pending = 0
    track.append(mido.MetaMessage("end_of_track", time=pending))
    smf0 = mido.MidiFile(type=0, ticks_per_beat=src.ticks_per_beat)
    smf0.tracks.append(track)
    smf0.save(str(out))


def midi_response(workdir: Path, src: Path, title: str, s: Settings) -> FileResponse:
    pm, counts = convert(src, workdir, s)
    out = workdir / "out.mid"
    write_smf(pm, out, s.smf_format)
    suffix = ("_parts" if s.separate else "") + ("_smf0" if s.smf_format == 0 else "")
    return FileResponse(
        out,
        media_type="audio/midi",
        filename=f"{safe_name(title)}{suffix}.mid",
        headers={
            "X-Note-Count": str(sum(counts.values())),
            "X-Part-Counts": ",".join(f"{k}:{v}" for k, v in counts.items()),
            "Access-Control-Expose-Headers": "X-Note-Count, X-Part-Counts, Content-Disposition",
        },
        background=BackgroundTask(shutil.rmtree, workdir, ignore_errors=True),
    )


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name).strip(" .")
    return (name or "converted")[:80]


def new_workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="midiconv_"))


def read_settings(onset, frame, min_note_ms, min_freq, max_freq, tempo, separate, parts, smf_format=1) -> Settings:
    return Settings(
        onset=onset, frame=frame, min_note_ms=min_note_ms,
        min_freq=min_freq or None, max_freq=max_freq or None, tempo=tempo,
        separate=separate,
        smf_format=0 if smf_format == 0 else 1,
        parts=[p for p in (parts or "").split(",") if p in PARTS] if parts else PART_ORDER,
    )


# ---------- API ----------

@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


def save_upload(file: UploadFile, workdir: Path) -> Path:
    src = workdir / f"input{Path(file.filename or '').suffix or '.bin'}"
    size = 0
    with src.open("wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"ファイルが大きすぎます（上限 {MAX_UPLOAD_MB}MB）")
            f.write(chunk)
    return src


YOUTUBE_RE = re.compile(r"^https?://(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/", re.I)


def download_youtube(url: str, workdir: Path) -> tuple[Path, str]:
    """YouTube から音声だけを取り出す。戻り値は (ファイル, 曲名)。"""
    import yt_dlp

    url = url.strip()
    if not YOUTUBE_RE.match(url):
        raise HTTPException(400, "YouTube のリンク（youtube.com / youtu.be）を入力してください。")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "yt.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("_type") == "playlist":
                raise HTTPException(400, "プレイリストではなく1曲のリンクを入力してください。")
            if (info.get("duration") or 0) > MAX_YOUTUBE_SECONDS:
                raise HTTPException(400, f"動画が長すぎます（上限 {MAX_YOUTUBE_SECONDS // 60} 分）")
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, f"YouTube から取得できませんでした: {str(e)[:200]}")
    downloaded = next(workdir.glob("yt.*"), None)
    if not downloaded:
        raise HTTPException(500, "音声のダウンロードに失敗しました。")
    return downloaded, info.get("title") or "youtube"


@app.post("/api/convert/file")
def convert_file(
    file: UploadFile = File(...),
    onset: float = Form(0.5), frame: float = Form(0.3), min_note_ms: float = Form(120),
    min_freq: float = Form(0), max_freq: float = Form(0), tempo: float = Form(120),
    separate: bool = Form(False), parts: str = Form(""), smf_format: int = Form(1),
):
    s = read_settings(onset, frame, min_note_ms, min_freq, max_freq, tempo, separate, parts, smf_format)
    workdir = new_workdir()
    try:
        src = save_upload(file, workdir)
        return midi_response(workdir, src, Path(file.filename or "converted").stem, s)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


@app.post("/api/convert/youtube")
def convert_youtube(
    url: str = Form(...),
    onset: float = Form(0.5), frame: float = Form(0.3), min_note_ms: float = Form(120),
    min_freq: float = Form(0), max_freq: float = Form(0), tempo: float = Form(120),
    separate: bool = Form(False), parts: str = Form(""), smf_format: int = Form(1),
):
    s = read_settings(onset, frame, min_note_ms, min_freq, max_freq, tempo, separate, parts, smf_format)
    workdir = new_workdir()
    try:
        src, title = download_youtube(url, workdir)
        return midi_response(workdir, src, title, s)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


# ---------- 音源解析（PSG / FM / 波形メモリ / PCM） ----------

def analyze_response(src: Path, workdir: Path, title: str) -> dict:
    import analyze

    wav = workdir / "analyze.wav"
    ffmpeg(src, wav, analyze.SR, 1)
    try:
        result = analyze.analyze(str(wav))
    except ValueError as e:
        raise HTTPException(400, str(e))
    result["title"] = title
    return result


@app.post("/api/analyze/file")
def analyze_file(file: UploadFile = File(...)):
    workdir = new_workdir()
    try:
        src = save_upload(file, workdir)
        return analyze_response(src, workdir, Path(file.filename or "audio").stem)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/api/analyze/youtube")
def analyze_youtube(url: str = Form(...)):
    workdir = new_workdir()
    try:
        src, title = download_youtube(url, workdir)
        return analyze_response(src, workdir, title)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    print("\n  ブラウザで http://127.0.0.1:8000 を開いてください\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
