# Raster Vision fastai Plugin

This plugin uses fastai and PyTorch to implement a semantic segmentation backend plugin.

## Setup and Requirements

### Docker
You'll need `docker` (preferably version 18 or above) installed. After cloning this repo, to build the Docker images, run the following command:

```shell
> docker/build
```

Before running the container, set an environment variable to a local directory in which to store data.
```shell
> export RASTER_VISION_DATA_DIR="/path/to/data"
```
To run a Bash console in the Docker container, invoke:
```shell
> docker/run
```
This will mount the following local directories to directories inside the container:
* `$RASTER_VISION_DATA_DIR -> /opt/data/`
* `fastai_plugin/ -> /opt/src/fastai_plugin/`
* `examples/ -> /opt/src/examples/`
* `scripts/ -> /opt/src/scripts/`

This script also has options for forwarding AWS credentials (`--aws`), running Jupyter notebooks (`--jupyter`), running on a GPU (`--gpu`), and others which can be seen below.

```
> ./docker/run --help
Usage: run <options> <command>
Run a console in the raster-vision-fastai Docker image locally.

Environment variables:
RASTER_VISION_DATA_DIR (directory for storing data; mounted to /opt/data)
AWS_PROFILE (optional AWS profile)
RASTER_VISION_REPO (optional path to main RV repo; mounted to /opt/src)

Options:
--aws forwards AWS credentials (sets AWS_PROFILE env var and mounts ~/.aws to /root/.aws)
--tensorboard maps port 6006
--gpu use the NVIDIA runtime and GPU image
--name sets the name of the running container
--jupyter forwards port 8888, mounts ./notebooks to /opt/notebooks, and runs Jupyter

All arguments after above options are passed to 'docker run'.
```

### Debug Mode

For debugging, it can be helpful to use a local copy of the Raster Vision source code rather than the version baked into the default Docker image. To do this, you can set the `RASTER_VISION_REPO` environment variable to the location of the main repo on your local filesystem. If this is set `docker/run` will mount `$RASTER_VISION_REPO/rastervision` to `/opt/src/rastervision` inside the container. You can then set breakpoints in your local copy of Raster Vision in order to debug experiments running inside the container.

## Running an experiment

To test the plugin, there is an [experiment](examples/vegas_buildings.py) using the SpaceNet Vegas buildings dataset. A test run can be executed locally using something like:
```
export RAW_URI="/opt/data/raw-data/spacenet-dataset"
export ROOT_URI="/opt/data/fastai/simple-seg/local-output/"
rastervision -p fastai run local -e examples.vegas_buildings \
    -a raw_uri $RAW_URI -a root_uri $ROOT_URI \
    -a test True --splits 2
```

A full remote run can be executed using something like:
```
export RAW_URI="s3://spacenet-dataset"
export ROOT_URI="s3://raster-vision-lf-dev/fastai/simple-seg/remote-output/"
rastervision -p fastai run aws_batch -e examples.vegas_buildings \
    -a raw_uri $RAW_URI -a root_uri $ROOT_URI \
    -a test False --splits 8
```

The `-p fastai` flag is to use a Raster Vision profile called `fastai`. This profile points to the location of the plugin module. You can make such a profile by creating a file at `~/.rastervision/fastai` containing something like:
```
[AWS_BATCH]
job_queue=lewfishRasterVisionGpuJobQueue
job_definition=lewfishRasterVisionCustomGpuJobDefinition
cpu_job_queue=lewfishRasterVisionCpuJobQueue
cpu_job_definition=lewfishRasterVisionCustomCpuJobDefinition
attempts=5

[AWS_S3]
requester_pays=True

[PLUGINS]
files=[]
modules=["fastai_plugin.semantic_segmentation_backend_config"]
```