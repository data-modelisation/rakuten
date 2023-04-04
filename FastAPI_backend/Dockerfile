# setup stage
FROM python:3.9-slim as builder

# set the working directory in the container 
WORKDIR /app

# python won’t try to write .pyc files on the import of source modules.
ENV PYTHONDONTWRITEBYTECODE 1
# Force the stdout and stderr streams to be unbuffered
ENV PYTHONUNBUFFERED 1

# need build-essential because fasttext required a compiler 
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip3 install --upgrade pip
   
# copy the dependences file to the working directory and install dependencies
COPY requirements.txt .
RUN pip3 wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r app/requirements.txt


# final stage
FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/*

#copy the content of the local directory 
COPY . /app

EXPOSE 8000
# command to run on container start 
ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "8000"]
CMD ["main:app",  "--reload"]