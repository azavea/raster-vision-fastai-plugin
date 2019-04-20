# Raster Vision fastai Plugin

This plugin uses fastai and PyTorch to implement a semantic segmentation backend plugin.

## Setup and Requirements

### Docker
You'll need `docker` (preferably version 18 or above) installed. After cloning this repo, to build the Docker images, run the following command:

```shell
> docker/build
```

This will pull down the latest `raster-vision:cpu-latest` Docker image and add some of this repo's code to it. Before running the container, set an environment variable to a local directory in which to store data.
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

For debugging, it can be helpful to use a local copy of the Raster Vision source code rather than the version baked into the default Docker image. To do this, you can set the `RASTER_VISION_REPO` environment variable to the location of the main repo on your local filesystem. If this is set, `docker/build` will set the base image to `raster-vision-cpu`, and `docker/run` will mount `$RASTER_VISION_REPO/rastervision` to `/opt/src/rastervision` inside the container. You can then set breakpoints in your local copy of Raster Vision in order to debug experiments running inside the container.
