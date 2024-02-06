#Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
# Set the working directory in the container to /app
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY requirements.txt .
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
# RUN apt-get install -y libglib2.0-0