# mri-modality-conversion

## Docker setup

### Install Docker

Download and install Docker (https://www.docker.com/).

### Run source code

Run following commands in the source code directory.

Build the docker image:
```
docker build -t mri-modality-conversion .
```

Execute the main file within the docker image:
```
docker run mri-modality-conversion python3 /code/pix2pix.py <arguments>
```

Arguments could be for example:
```
docker run mri-modality-conversion python3 /code/pix2pix.py -f data -m convert -c t1_to_t2
```
