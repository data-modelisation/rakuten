FROM python:3.9.13-slim

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 8080

# ENTRYPOINT ["streamlit","run", "--server.port", 8080, "--server.enableCORS", false]
# CMD ["app.py"]

CMD streamlit run --server.port 8080 --server.enableCORS false app.py