FROM python:3.11-slim

WORKDIR /app

# system deps for some packages (optional, add as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 8501

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/bin/bash", "./start.sh"]
