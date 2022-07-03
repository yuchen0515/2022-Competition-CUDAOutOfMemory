FROM python:3.7.6-slim

WORKDIR /app

COPY requirements.txt ./
RUN \
  pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "./api.py"]
