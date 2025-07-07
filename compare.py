from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
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
  #"n_heads": 8,
  "d_head": 64,
  "transformer_depth": [1, 1, 1, 1],
  #"ctx_dim": 768,
  "ctx_dim": 1024,
  #"use_linear": False,
  "use_linear": True,
}
model = UNetModel(**unet_params)

# from mlperf reference implementation
unet_weights = safe_load("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/unet_training_init_model.safetensors")
unet_io = safe_load("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/unet_training_io.safetensors")
for k,v in unet_io.items():
  unet_io[k] = v.to(Device.DEFAULT).realize()

load_state_dict(model, unet_weights)

out = model(unet_io['x'], unet_io['timesteps'], unet_io['context']).realize()

def md(a, b):
  diff = (a - b).abs().mean().item()
  ratio = diff / a.abs().mean().item()
  return diff, ratio

diff, ratio = md(unet_io['out'], out)

end = 1