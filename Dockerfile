# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY app.py /code/app.py

# 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
