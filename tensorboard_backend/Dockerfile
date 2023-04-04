FROM tensorflow/tensorflow:latest

WORKDIR /app

EXPOSE 6009

ENTRYPOINT ["tensorboard", "--host", "0.0.0.0", "--port", "6009"]
CMD ["--logdir", "/app/text/text_neural_simple/layers"]
