# Single-container deployment (Hugging Face Spaces): FastAPI on internal :8000,
# Streamlit public on :7860. One container rather than two services so the API
# never needs a public URL -- it binds loopback, and the only way to reach it is
# through the Streamlit app, which sits behind the password gate.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Unprivileged account that agent-generated code is dropped to. It owns nothing
# except the workspace tree, so it cannot read app.db, the uploads, or the source.
RUN useradd --create-home --uid 1001 --shell /usr/sbin/nologin sandbox

COPY backend/requirements.txt backend/requirements.txt
COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt -r frontend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY start.sh /start.sh
RUN chmod +x /start.sh

ENV SANDBOX_BACKEND=subprocess \
    SANDBOX_RUN_AS_USER=sandbox \
    SANDBOX_TIMEOUT_SECONDS=90 \
    DATA_DIR=/app/data \
    WORKSPACES_DIR=/sandbox/workspaces \
    BACKEND_HOST=127.0.0.1 \
    BACKEND_PORT=8000 \
    BACKEND_BASE_URL=http://127.0.0.1:8000 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 700 on the app's data dir is the point of the split: the sandbox user cannot
# even traverse into it, so app.db and the raw uploads are out of reach. Its own
# workspace tree lives elsewhere and is writable.
RUN mkdir -p /app/data/uploads /sandbox/workspaces \
 && chmod 700 /app/data \
 && chown -R sandbox:sandbox /sandbox \
 && chmod 700 /sandbox/workspaces

EXPOSE 7860

CMD ["/start.sh"]
