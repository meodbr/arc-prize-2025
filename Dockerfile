FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310
USER root

# Restrain user privileges
WORKDIR /opt/app

RUN pip install --no-cache-dir --upgrade pip setuptools>=60 wheel
ENV SETUPTOOLS_USE_DISTUTILS=local

# Install dependencies
COPY pyproject.toml ./pyproject.toml

# Add sources and install project
RUN touch README.md
COPY src ./src
RUN pip install --no-cache-dir .
RUN rm -r /opt/python/3.10/lib/python3.10/distutils

ENTRYPOINT ["python"]
CMD ["-m", "arc_tartiflette.train"]