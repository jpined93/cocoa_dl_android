FROM python:3.9.13
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --user
EXPOSE 5000
CMD [ "python3","-m","flask","run","--host=0.0.0.0" ]