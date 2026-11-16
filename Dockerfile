FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN python -m pip install --no-cache-dir numpy==2.4.6 Pillow==13.0.0 \
    && python -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.10.0 torchvision==0.25.0

COPY predict.py web_demo.py ./
COPY models/ ./models/
COPY utils/ ./utils/
COPY web/ ./web/
COPY splits/test.json ./splits/test.json
COPY resnet34_full/training_history.json resnet34_full/val_score.json resnet34_full/test_score.json ./resnet34_full/

USER 65534:65534
EXPOSE 7861
CMD ["python", "web_demo.py", "--host", "0.0.0.0", "--device", "cpu"]
