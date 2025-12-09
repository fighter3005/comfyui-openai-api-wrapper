FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# CMD ["python", "app.py"]
CMD ["gunicorn", "-w", "4", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:5000", "app:app"]