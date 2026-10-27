# The MBT-A editing sandbox for the E2B backend.
#   cd docker && e2b template build    (reads e2b.toml: 1 CPU, 2048 MB)
FROM e2bdev/base:latest

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && (command -v npm >/dev/null 2>&1 || apt-get install -y --no-install-recommends nodejs npm) \
 && rm -rf /var/lib/apt/lists/*

ARG CLAUDE_CODE_VERSION=2.1.283
RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"

ENV DISABLE_AUTOUPDATER=1 \
    DISABLE_TELEMETRY=1 \
    DISABLE_ERROR_REPORTING=1
