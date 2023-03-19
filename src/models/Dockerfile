FROM tensorflow/tensorflow:latest

WORKDIR /app

EXPOSE 6009

COPY text/text_neural_simple/layers /text/text_neural_simple/layers

CMD tensorboard --logdir /app/text/text_neural_simple/layers --host 127.0.0.1 --port 6009