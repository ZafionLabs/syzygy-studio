FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y nodejs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt package.json package-lock.json ./
RUN pip install --no-cache-dir -r requirements.txt && npm install

COPY . .
RUN npm run build

ENV PYTHON_URL=http://127.0.0.1:8010
ENV PORT=8080
EXPOSE 8080

CMD python -m terrain_bot serve --host 127.0.0.1 --port 8010 & \
    i=0; \
    while [ "$i" -lt 60 ]; do \
      python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/api/info')" && break; \
      i=$((i+1)); sleep 2; \
    done; \
    exec npx tsx server/index.ts
