# set base image
FROM python:3.9

# copy the dependences file to the working directory
COPY requirements.txt app/requirements.txt
#set the working directory in the container 
WORKDIR /app

# RUN apt-get update &&\
#     apt-get install --no-install-recommends --yes build-essential
RUN pip3 install --upgrade pip
# install dependencies
RUN pip3 install -r requirements.txt
#copy the content of the local directory 
COPY . /app

EXPOSE 8000
# command to run on container start 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" , "--reload"]
