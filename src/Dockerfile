FROM python:3.9.13-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update &&\
    apt-get install --no-install-recommends --yes build-essential
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8008

CMD uvicorn fast:app --host 127.0.0.1 --port 8008 --reload