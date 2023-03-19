FROM tensorflow/tensorflow:latest

WORKDIR /app

EXPOSE 6009

COPY text/text_neural_simple/layers/checkpoint text/text_neural_simple/layers/checkpoint 
COPY text/text_neural_simple/layers/embedding.ckpt-1.data-00000-of-00001 text/text_neural_simple/layers/embedding.ckpt-1.data-00000-of-00001 
COPY text/text_neural_simple/layers/embedding.ckpt-1.index text/text_neural_simple/layers/embedding.ckpt-1.index 

CMD tensorboard --logdir /app/text/text_neural_simple/layers --host 127.0.0.1 --port 6009