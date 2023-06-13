FROM python:3.8.10

WORKDIR home/

RUN mkdir results

COPY requirements.txt requirements.txt
# Need to have tefex installed inside seventh sensic repo

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

RUN pip freeze

COPY simplified/ simplified/
COPY Data/ Data/

CMD ["simplified/masked_ppo.py"]
ENTRYPOINT ["python"]
