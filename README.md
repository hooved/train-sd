```shell
git submodule add https://github.com/tinygrad/tinygrad.git tinygrad-repo && \
cd tinygrad-repo && \
pip install -e .
```

```shell
git submodule add https://github.com/mlcommons/training.git training

# NOTE: this downloads ~1 TB
./training/stable_diffusion/scripts/datasets/laion400m-filtered-download-moments.sh --output-dir ./datasets/laion-400m/webdataset-moments-filtered

./training/stable_diffusion/scripts/datasets/coco2014-validation-download-prompts.sh --output-dir ./datasets/coco2014
./training/stable_diffusion/scripts/datasets/coco2014-validation-download-stats.sh --output-dir ./datasets/coco2014

# This downloads ~5 GB
./training/stable_diffusion/scripts/checkpoints/download_sd.sh --output-dir ./checkpoints/sd

./training/stable_diffusion/scripts/checkpoints/download_inception.sh --output-dir ./checkpoints/inception

# This downloads 3.7 GB
./training/stable_diffusion/scripts/checkpoints/download_clip.sh --output-dir ./checkpoints/clip

# should have used training/stable_diffusion as output path
mv datasets checkpoints training/stable_diffusion/
```

```shell
cd training/stable_diffusion && \
docker build -t mlperf/stable_diffusion .
```

```shell
mkdir results && \
docker run --rm -it --gpus=all --ipc=host \
  --workdir /pwd \
  -v ${PWD}:/pwd \
  -v ${PWD}/datasets/laion-400m:/datasets/laion-400m \
  -v ${PWD}/datasets/coco2014:/datasets/coco2014 \
  -v ${PWD}/checkpoints:/checkpoints \
  -v ${PWD}/results:/results \
  mlperf/stable_diffusion bash
```

```shell
# within container
./run_and_time.sh \
  --num-nodes 1 \
  --gpus-per-node 1 \
  --checkpoint /checkpoints/sd/512-base-ema.ckpt \
  --results-dir /results \
  --config configs/train_01x01x01.yaml
```

# VSCode debugging in container
```shell
# important: first build docker image as above, then ensure this config is present:
# training/stable_diffusion/.devcontainer/devcontainer.json

# within vscode: Dev Containers: open folder in container
# if there's an issue, edit the devcontainer.json and vscode>dev containers: rebuild image
# check hostname within container matches docker ps -a of correct image
```
