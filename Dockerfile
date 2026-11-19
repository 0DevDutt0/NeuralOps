# Build the package and its runtime dependencies into a virtual environment.
FROM python:3.11-slim AS builder
WORKDIR /app
ARG INSTALL_GUARDRAILS=0
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY pyproject.toml README.md LICENSE ./
COPY neuralops ./neuralops
RUN if [ "$INSTALL_GUARDRAILS" = "1" ]; then \
        pip install --no-cache-dir '.[guardrails]' && python -m spacy download en_core_web_lg; \
    else \
        pip install --no-cache-dir .; \
    fi

FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PATH="/opt/venv/bin:${PATH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=builder /opt/venv /opt/venv
COPY main.py alembic.ini ./
COPY neuralops ./neuralops
COPY dashboard ./dashboard
RUN useradd -m -u 1000 neuralops && chown -R neuralops:neuralops /app
USER neuralops
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=5).close()"]
EXPOSE 8000
CMD ["python", "main.py"]
