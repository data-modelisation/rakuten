FROM python:3.9.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

RUN addgroup --system app && adduser --system --group app
USER app

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "--server.port", "8080", "--server.enableCORS", "false"]
CMD ["app.py"]

