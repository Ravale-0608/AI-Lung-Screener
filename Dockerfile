# ── Lung Risk Screener — API image ────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps required by SimpleITK / PyRadiomics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch + lungmask (keeps image ~2 GB lighter than CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir lungmask

COPY ./app  ./app
COPY ./model ./model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
