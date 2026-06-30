# demo/Dockerfile — Cloud Run image for the Streamlit demo app.
#
# Standalone — does not need the Temporal worker, Postgres, or Redis.
# Runs in simulation mode with no API keys; set GROQ_API_KEY / OPENAI_API_KEY /
# ANTHROPIC_API_KEY as Cloud Run secrets to enable real LLM calls.
#
# Build locally:
#   docker build -f demo/Dockerfile -t oil-price-demo-ui .
#   docker run -p 8080:8080 oil-price-demo-ui
#
# Cloud Run (manual one-off):
#   gcloud run deploy oil-price-demo-ui \
#     --source . --dockerfile demo/Dockerfile \
#     --region us-central1 --project $GCP_PROJECT_ID \
#     --allow-unauthenticated --port 8080

FROM python:3.12-slim

WORKDIR /app

COPY demo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY demo/app.py /app/app.py

# Cloud Run injects $PORT (default 8080); Streamlit must bind to it.
ENV PORT=8080

EXPOSE 8080

# --server.headless=true    suppress browser-open + CORS warnings
# --server.enableCORS=false not needed for Cloud Run (no cross-origin iframe)
# --server.address=0.0.0.0  listen on all interfaces, not just localhost
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
