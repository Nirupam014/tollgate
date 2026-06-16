# Tollgate container image — for CI systems that prefer running a container.
#
#   docker build -t tollgate .
#   docker run --rm -v "$PWD:/repo" tollgate analyze /repo --fail-on block
#
# Includes the multilang extra so Go/Java/Ruby get full graph recovery.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Tollgate" \
      org.opencontainers.image.description="Prevention-first token-risk analysis for AI agents, for CI/CD." \
      org.opencontainers.image.source="https://github.com/Nirupam014/tollgate" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[multilang]"

# Drop into a non-root user; the scanned repo is mounted read-only at /repo.
RUN useradd --create-home --uid 10001 tollgate
USER tollgate
WORKDIR /repo

ENTRYPOINT ["tollgate"]
CMD ["--help"]
