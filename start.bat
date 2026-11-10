@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  echo 初回セットアップ中...
  py -3.11 -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
)
start "" http://127.0.0.1:8000
.venv\Scripts\python app.py
