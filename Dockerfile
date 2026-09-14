FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY final_score.py .
COPY validate_submission.py .

CMD ["python3", "final_score.py"]
