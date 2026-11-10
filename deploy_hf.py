"""Hugging Face Spaces に公開（更新）するスクリプト。

事前に一度だけ `hf auth login` で書き込み権限のあるトークンでログインしておきます。
使い方:  .venv\\Scripts\\python deploy_hf.py            （Space 名は YouTube-SMF）
        .venv\\Scripts\\python deploy_hf.py 別の名前
"""

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent
FILES = ["app.py", "analyze.py", "requirements.txt", "Dockerfile", ".dockerignore", "static/index.html"]

# Spaces の設定（README の先頭に書く決まり）
SPACE_HEADER = """---
title: YouTube-SMF
emoji: 🎹
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: 音声・YouTubeからMIDI(SMF)変換と音源解析
---

"""


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "YouTube-SMF"
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in FILES:
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, tmp / f)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        (tmp / "README.md").write_text(SPACE_HEADER + readme, encoding="utf-8")
        api.upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="space",
                          commit_message="Update app")

    print(f"アップロードしました: https://huggingface.co/spaces/{repo_id}")
    print("ビルドに10〜20分ほどかかります。")


if __name__ == "__main__":
    main()
