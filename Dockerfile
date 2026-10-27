FROM python:3.12-slim-bullseye

WORKDIR /app


COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py model.py

ENTRYPOINT ["python", "model.py"] 

EXPOSE 8080