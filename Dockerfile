FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (if any) – none needed for pure Python

# Copy only requirements first for caching layers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code and artefacts
COPY . /app

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

