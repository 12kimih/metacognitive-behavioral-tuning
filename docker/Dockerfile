# The MBT-A editing sandbox for the Docker backend: Node.js, git and a pinned Claude Code.
#   docker build -t mbt-editor:latest -f docker/Dockerfile docker
FROM node:22-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates bash coreutils \
 && rm -rf /var/lib/apt/lists/*

# The paper used Claude Code 2.1.283.
ARG CLAUDE_CODE_VERSION=2.1.283
RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
 && npm cache clean --force

ENV DISABLE_AUTOUPDATER=1 \
    DISABLE_TELEMETRY=1 \
    DISABLE_ERROR_REPORTING=1
WORKDIR /workspace
