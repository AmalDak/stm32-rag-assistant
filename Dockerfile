FROM python:3.11-slim

WORKDIR /app

# Pin numpy first to avoid version conflicts
RUN pip install --no-cache-dir "numpy<2"

# Install CPU-only torch that works with latest sentence-transformers
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]