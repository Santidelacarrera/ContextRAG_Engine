FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY contextrag ./contextrag
# add ,models for bge / cross-encoder / DeBERTa, ,anthropic for Claude (large images)
RUN pip install --no-cache-dir ".[api,qdrant,postgres,redis,security,pdf,memory]"
# do not run as root: a compromise of the parser or the API then has no write access to the image or the host's files
RUN useradd --system --uid 10001 --no-create-home app && mkdir /data && chown app /data
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
CMD ["uvicorn", "--factory", "contextrag.api.server:app_factory", "--host", "0.0.0.0", "--port", "8000"]
