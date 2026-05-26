FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY packages ./packages
COPY services/analysis-adapter ./service
RUN pip install --no-cache-dir -e .
WORKDIR /app/service
ENV PYTHONPATH=/app/packages:/app/service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8085"]
