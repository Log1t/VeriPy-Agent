FROM python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217
COPY docker/python-sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check --no-compile -r /tmp/requirements.txt
RUN useradd --create-home --uid 10001 worker
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install --no-cache-dir --disable-pip-version-check --no-deps /app
COPY docker/python-sandbox/worker.py /app/worker.py
USER worker
ENTRYPOINT ["python", "-I", "/app/worker.py"]
