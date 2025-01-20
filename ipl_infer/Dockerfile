FROM python:3.11.9-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

ENV MODEL_DIR=/app/MODEL

RUN mkdir -p /app/MODEL
COPY ["predict_ws.py", "./"]
COPY ["MODEL/ipl_chase_pred_v1.bin", "/app/MODEL/"]


EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_ws:app"]