# Use official lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    google-genai \
    "anthropic[vertex]"

# Copy source code
COPY main.py /app/main.py
COPY config.py /app/config.py
COPY models.py /app/models.py
COPY handlers/ /app/handlers/
COPY converters/ /app/converters/

# Environment variable configuration
# Cloud Run automatically sets GOOGLE_CLOUD_PROJECT
# GOOGLE_CLOUD_LOCATION defaults to global
ENV GOOGLE_CLOUD_LOCATION=global
ENV GOOGLE_GENAI_USE_VERTEXAI=true

# Expose port
EXPOSE 8080

# Start command
CMD ["python", "main.py"]