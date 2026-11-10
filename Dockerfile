# Hugging Face Spaces（Docker）用。他の Docker ホスティングでも動きます。
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Spaces は UID 1000 のユーザーで動かす決まり
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HOST=0.0.0.0 \
    PORT=7860
WORKDIR /home/user/app

# PyTorch は GPU 版だと数GBあるので CPU 版を先に入れる
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# パート分けの AI モデル（htdemucs）を先にダウンロードしておく
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
