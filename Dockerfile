FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
