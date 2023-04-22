FROM python:3.9

WORKDIR /cars

COPY ./requirements.txt /cars/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /cars/requirements.txt

COPY . /cars

CMD ["/bin/bash", "-c", "python -m src.train"]