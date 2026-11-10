# YouTube-SMF

MP3・WAV などの音声ファイル、または YouTube の曲のリンクから、**MIDI ファイル（SMF 形式 .mid）** を作る Web ツールです。

## できること

- **音声ファイルから変換**：MP3 / WAV / M4A / FLAC / OGG など
- **YouTube のリンクから変換**：URL を貼るだけ（1曲15分まで）
- **パート分け**：AI で「ボーカル・その他の伴奏・ベース・ドラム」に分けて、パートごとに別トラックにする
- **SMF フォーマット0 / 1 を選べる**
  - フォーマット1：パートごとに別トラック（DAW・作曲ソフト向け）
  - フォーマット0：1トラックにまとめる（古いキーボード・カラオケ・ゲーム機向け）
- **プレビュー**：ピアノロール表示とブラウザでの試聴、パートごとの .mid ダウンロード
- **詳しい設定**：検出感度、最短の音の長さ、テンポ、音域

## 必要なもの

- Windows（Mac / Linux でも `app.py` は動きます）
- Python 3.11
- [ffmpeg](https://ffmpeg.org/)（PATH が通っていること）
  - Windows なら `winget install ffmpeg` で入ります

## 使い方

### Windows

`start.bat` をダブルクリックします。初回は必要なライブラリを自動でインストールします（数分かかります）。
起動するとブラウザで http://127.0.0.1:8000 が開きます。

### 手動で起動する場合

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Mac/Linux は .venv/bin/python
.venv/Scripts/python app.py
```

ブラウザで http://127.0.0.1:8000 を開いてください。

## しくみ

| 処理 | 使っているもの |
|---|---|
| 音声 → ノート（音程のある楽器） | [basic-pitch](https://github.com/spotify/basic-pitch)（Spotify） |
| パート分け | [Demucs](https://github.com/facebookresearch/demucs)（htdemucs） |
| ドラム | 帯域ごとの立ち上がり検出でキック / スネア / ハイハットに打ち込み |
| YouTube の音声取得 | [yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| サーバー | FastAPI |
| プレビュー | @tonejs/midi、Tone.js |

## 注意

- 自動採譜なので、完全に正確な楽譜にはなりません。ピアノ独奏や歌のメロディなど、シンプルな曲ほどきれいに取れます。
- パート分けは CPU で処理するため、曲の長さによって数分かかります。
- ドラムはキック・スネア・ハイハットの3種類のみです。
- **YouTube からの音声取得は、YouTube の利用規約に反する場合があります。自分で権利を持っている曲や、利用が許可された曲だけに使ってください。** 変換したデータの扱いは利用者の責任でお願いします。
