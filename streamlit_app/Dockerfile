FROM python:3.9.13-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8080

CMD streamlit run --server.port 8080 --server.enableCORS false app.py