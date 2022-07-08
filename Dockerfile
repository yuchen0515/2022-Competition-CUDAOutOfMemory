FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /app

COPY requirements.txt ./
RUN \
  pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "./api.py"]
