FROM python:3.8.10

COPY . /app
WORKDIR /app

RUN pip install pipenv
RUN pipenv lock -r > requirements.txt
# deploy latest Pipfile.lock to virtual system # --deploy 
RUN pip3 install -r requirements.txt

# Make port vailable for links and/or publish
# EXPOSE 8003 # do at runtime via run

ENTRYPOINT ["./app_starter.sh"]