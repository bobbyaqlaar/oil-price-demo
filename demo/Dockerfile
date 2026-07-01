# demo/Dockerfile — Cloud Run image for the Streamlit demo GUI.
#
# Connects to the live Temporal server and Temporal worker — does not embed
# the worker itself. Set TEMPORAL_ADDRESS / TEMPORAL_TLS / TENANT_ID at
# deploy time (Cloud Run env vars or --set-env-vars).
#
# Build:
#   docker build -f demo/Dockerfile -t oil-price-demo-ui .
#   docker run -p 8080:8080 -e TEMPORAL_ADDRESS=localhost:7233 oil-price-demo-ui
#
# Cloud Run deploy (manual one-off):
#   gcloud run deploy oil-price-demo-ui \
#     --source . --dockerfile demo/Dockerfile \
#     --region us-central1 --project $GCP_PROJECT_ID \
#     --allow-unauthenticated --port 8080 \
#     --set-env-vars TEMPORAL_ADDRESS=<host>:7233,TENANT_ID=oil-price-demo

FROM python:3.12-slim

WORKDIR /app

COPY demo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY demo/app.py /app/app.py

ENV PORT=8080

EXPOSE 8080

# hadolint ignore=DL3025
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
