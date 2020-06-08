FROM jupyter/minimal-notebook:c76996e26e48

LABEL maintainer="Chia Yew Ken <yewken_chia@mymail.sutd.edu.sg>"

USER root

#ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

ENV FILE="requirements.txt"
COPY $FILE $FILE
RUN pip install -r $FILE

#ENV FILE="get_data.sh"
#COPY $FILE $FILE
#RUN bash $FILE

#EXPOSE 5000

#ENTRYPOINT [ "python3" ]
#CMD [ "app.py" ]
