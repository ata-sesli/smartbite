FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG SMARTBITE_INSTALL_AI=0

COPY pyproject.toml README.md ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini main.py ./

RUN if [ "$SMARTBITE_INSTALL_AI" = "1" ]; then \
      pip install --no-cache-dir ".[ai]"; \
    else \
      pip install --no-cache-dir .; \
    fi

CMD ["smartbite-api"]
