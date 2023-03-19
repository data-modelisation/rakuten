FROM tensorflow/tensorflow:latest

WORKDIR /app

EXPOSE 6009

# COPY models/checkpoint models/checkpoint
# COPY models/embedding.ckpt-1.data-00000-of-00001 models/embedding.ckpt-1.data-00000-of-00001
# COPY models/embedding.ckpt-1.index models/embedding.ckpt-1.index
# COPY models/metadata.tsv models/metadata.tsv

CMD tensorboard --logdir /app/models --host 127.0.0.1 --port 6009