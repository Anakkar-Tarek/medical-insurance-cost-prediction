FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 5000

CMD ["python", "app.py"]
