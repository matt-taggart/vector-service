# Use the official Python image as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the Flask app code to the working directory
COPY src/app.py .

# Expose the port that the Flask app will be listening on
EXPOSE 5000

# Set the environment variable to run the Flask app
ENV FLASK_APP=app.py

# Set the command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
