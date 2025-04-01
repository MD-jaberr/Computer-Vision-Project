FROM python:3.10-slim

RUN mkdir /app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy all files into the container
COPY . .

EXPOSE 8000

#run FastAPI with uvicorn
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
