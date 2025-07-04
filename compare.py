from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load, load_state_dict
from typing import Any

# goal: translate mlperf stable_diffusion-v2 ref. implementation to tinygrad

# not shown: generation of safetensors from mlperf reference implementation
# to get these safetensors, setup the mlperf docker image and config (see README), and export the safetensors where indicated in the commented mlperf code

### unet forward

from extra.models.unet import UNetModel

unet_params: dict[str,Any] = {
  "adm_in_ch": None,
  "in_ch": 4,
  "out_ch": 4,
  "model_ch": 320,
  "attention_resolutions": [4, 2, 1],
  "num_res_blocks": 2,
  "channel_mult": [1, 2, 4, 4],
  "n_heads": 8,
  "transformer_depth": [1, 1, 1, 1],
  "ctx_dim": 768,
  "use_linear": False,
}
model = UNetModel(**unet_params)

# from mlperf reference implementation
unet_weights = safe_load("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/unet_training_init_model.safetensors")
unet_io = safe_load("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/unet_training_forward.safetensors")

unet_weights['input_blocks.1.1.proj_in.weight'] = unet_weights['input_blocks.1.1.proj_in.weight'].unsqueeze(-1).unsqueeze(-1)

load_state_dict(model, unet_weights)

end = 1