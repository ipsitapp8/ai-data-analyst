FROM python:3.11-slim

# Minimal, pinned analysis stack for agent-generated code.
# No network access is granted at runtime (--network none), so packages
# must all be baked into the image at build time.
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    plotly==5.22.0 \
    matplotlib==3.8.4

# Run as a non-root, unprivileged user.
RUN useradd -m -u 1000 sandboxuser
USER sandboxuser

WORKDIR /workspace

ENTRYPOINT ["python"]
