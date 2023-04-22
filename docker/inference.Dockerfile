FROM python:3.9

WORKDIR /cars

COPY ./requirements.txt /cars/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /cars/requirements.txt

COPY . /cars

EXPOSE 4000
CMD ["uvicorn", "src.fastapi:app", "--host", "0.0.0.0", "--port", "4000"]