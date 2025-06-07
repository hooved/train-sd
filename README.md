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
```
