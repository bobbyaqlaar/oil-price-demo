# oil-price-demo/Dockerfile — container image for the Temporal worker
# (poller + /healthz). Build context is this repo's root. runtime/ here is
# a vendored copy of AgentSmith's runtime/ package (see OPERATIONS.md §D.5b
# / §2.2) — there's no shared framework checkout inside a container, so it
# gets baked into the image instead of resolved via $AGENTSMITH_DIR at
# runtime the way local dev does.
#
# Build:  docker build -t oil-price-worker .
# CI/CD:  .github/actions/build-push-ghcr picks this up automatically.

FROM python:3.12-slim

WORKDIR /app

COPY runtime/requirements-runtime.txt /app/requirements-runtime.txt
RUN pip install --no-cache-dir -r /app/requirements-runtime.txt

COPY runtime/ /app/runtime/
COPY . /app/

ENV AGENTSMITH_DIR=/app

EXPOSE 8080
CMD ["python3", "worker.py"]
