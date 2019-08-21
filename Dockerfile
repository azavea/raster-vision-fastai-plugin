FROM raster-vision-base:latest

RUN pip install tensorboardX==1.8 future==0.17.*

COPY ./fastai_plugin /opt/src/fastai_plugin
COPY ./examples /opt/src/examples

ENV PYTHONPATH /opt/src/:/opt/src/fastai:$PYTHONPATH

CMD ["bash"]
