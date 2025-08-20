FROM python:3.10-slim
USER root

# Restrain user privileges
WORKDIR /opt/app

# Setup configuration
ENV PATH="$PATH:/home/mmesi/.local/bin"
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1
ENV DOCLING_ARTIFACTS_PATH=/home/mmesi/.cache/docling/models

# Install uv
RUN pip install --upgrade --no-cache-dir uv==0.6.16

# Install dependencies
COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock
RUN uv sync --locked --no-dev --no-install-project --no-cache-dir

# Add sources and install project
RUN touch README.md
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --no-cache-dir

ENTRYPOINT ["uv", "run", "--no-dev", "--no-build", "--no-editable", "--no-sync", "arc-tartiflette"]
CMD ["python", "-m", "src.arc_tartiflette.train"]