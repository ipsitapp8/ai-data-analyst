FROM python:3.11-slim

# Minimal, pinned analysis stack for agent-generated code. Versions match
# backend/requirements.txt so a script behaves the same here as in the API's
# own profiling code. No network access is granted at runtime
# (--network none), so everything must be baked in at build time.
RUN pip install --no-cache-dir \
    pandas==2.3.3 \
    numpy==2.4.6 \
    plotly==6.9.0 \
    matplotlib==3.11.1

# Run as a non-root, unprivileged user. The mount points exist in the image
# because the root filesystem is read-only at runtime (--read-only).
RUN useradd -m -u 1000 sandboxuser \
 && mkdir -p /workspace/data /workspace/output \
 && chown -R sandboxuser:sandboxuser /workspace
USER sandboxuser
WORKDIR /workspace

ENTRYPOINT ["python"]
