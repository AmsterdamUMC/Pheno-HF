FROM python:3.11-slim

# Build tools needed by a few packages that compile from source
# (e.g. hnswlib, hdbscan) and by symspellpy's C extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
		git \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

EXPOSE 5678

CMD ["bash"]
