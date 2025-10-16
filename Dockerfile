FROM python:3.12.6-slim

RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /api

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
