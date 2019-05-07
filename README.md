# Raster Vision PyTorch/fastai Plugin

This plugin uses [PyTorch](https://pytorch.org/) and [fastai](https://docs.fast.ai/) to implement a semantic segmentation backend plugin for [Raster Vision](https://rastervision.io/).

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

This script also has options for forwarding AWS credentials (`--aws`), running Jupyter notebooks (`--jupyter`), running on a GPU (`--gpu`), and others.
Run `docker/run --help` for more details.

### Debug Mode

For debugging, it can be helpful to use a local copy of the Raster Vision source code rather than the version baked into the Docker image. To do this, you can set the `RASTER_VISION_REPO` environment variable to the location of the main repo on your local filesystem. If this is set, `docker/run` will mount `$RASTER_VISION_REPO/rastervision` to `/opt/src/rastervision` inside the container. You can then modify your local copy of Raster Vision in order to debug experiments running inside the container.

### (Optional) Setup AWS Batch

This assumes that a Batch stack was created using the [Raster Vision AWS Batch setup](https://github.com/azavea/raster-vision-aws).
To use this plugin, you will need to add a job definition which points to a new tag on the ECR repo, and then publish the image to that tag.
You can do this by editing [scripts/cpu_job_def.json](scripts/cpu_job_def.json), [scripts/gpu_job_def.json](scripts/gpu_job_def.json], and [docker/publish_image], and then running `docker/publish_image` outside the container, and `scripts/add_job_defs` inside the container.

### Setup profile

Using the plugin requires making a Raster Vision profile which points to the location of the plugin module. You can make such a profile by creating a file at `~/.rastervision/fastai` containing something like the following. If using Batch, the `AWS_BATCH` section should point to the resources created above.

```
[AWS_BATCH]
job_queue=lewfishRasterVisionGpuJobQueue
job_definition=lewfishFastaiPluginGpuJobDef
cpu_job_queue=lewfishRasterVisionCpuJobQueue
cpu_job_definition=lewfishFastaiPluginCpuJobDef
attempts=5

[AWS_S3]
requester_pays=False

[PLUGINS]
files=[]
modules=["fastai_plugin.semantic_segmentation_backend_config"]
```

## Running an experiment

To test the plugin, you can run an [experiment](examples/potsdam.py) using the ISPRS Potsdam dataset. Info on setting up the data and experiments in general can be found in the [examples repo](https://github.com/azavea/raster-vision-examples#isprs-potsdam-semantic-segmentation). A test run can be executed locally using something like the following. The `-p fastai` flag says to use the `fastai` profile created above.

```
export RAW_URI="/opt/data/raw-data/isprs-potsdam"
export PROCESSED_URI="/opt/data/fastai/potsdam/processed-data"
export ROOT_URI="/opt/data/fastai/potsdam/local-output"
rastervision -p fastai run local -e examples.semantic_segmentation.potsdam -m *exp_resnet50* \
    -a raw_uri $RAW_URI -a processed_uri $PROCESSED_URI -a root_uri $ROOT_URI \
    -a test True --splits 2
```

A full experiment can be run on AWS Batch using something like:

```
export RAW_URI="s3://raster-vision-raw-data/isprs-potsdam"
export PROCESSED_URI="s3://raster-vision-lf-dev/fastai/potsdam/processed-data"
export ROOT_URI="s3://raster-vision-lf-dev/fastai/potsdam/remote-output"
rastervision -p fastai run aws_batch -e examples.semantic_segmentation.potsdam -m *resnet18* \
    -a raw_uri $RAW_URI -a processed_uri $PROCESSED_URI -a root_uri $ROOT_URI \
    -a test False --splits 4
```

This gets to an average F1 score of 0.87 after 15 minutes of training.