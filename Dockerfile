# Use an official Python image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the app files
COPY app/ /app/
COPY models/ /app/models/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
