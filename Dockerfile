FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "cli.py"]
