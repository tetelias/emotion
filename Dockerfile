FROM python:3.10

# Change container workdir to /app
WORKDIR /app
COPY ./requirements.txt  /app/requirements.txt
# Install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy source code to container
COPY ./src /app

# CMD python serve.py
CMD uvicorn api:app --host 0.0.0.0