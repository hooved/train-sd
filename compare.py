from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from typing import Any
from collections import namedtuple

def md(a, b):
  diff = (a - b).abs()
  max_diff = diff.max().item()
  mean_diff = diff.mean().item()
  ratio = mean_diff / a.abs().mean().item()
  return mean_diff, ratio, max_diff

# goal: translate mlperf stable_diffusion-v2 ref. implementation to tinygrad

# not shown: generation of safetensors from mlperf reference implementation
# to get these safetensors, setup the mlperf docker image and config (see README), and export the safetensors where indicated in the commented mlperf code

# *** clip encoding closeness testing
from extra.models.clip import FrozenOpenClipEmbedder
data = safe_load("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/unet_inputs.safetensors")
for k,v in data.items():
  data[k] = v.to("NV").realize()

with open("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/cond.txt", encoding="utf-8") as f:
  prompt = f.read()

# For loading mlperf reference weights state_dict
class ClipContainer:
  def __init__(self):
    clip_config = {"dims": 1024, "n_heads": 16, "layers": 24, "return_pooled": False, "ln_penultimate": True}
    self.cond_stage_model = FrozenOpenClipEmbedder(**clip_config)

model = ClipContainer()
mlperf_training_init_model = safe_load("/home/hooved/train-sd/training/stable_diffusion/checkpoints/training_init_model.safetensors")
load_state_dict(model, mlperf_training_init_model)
context = model.cond_stage_model([prompt]).realize()

"""
md(data['c'], context)
(2.1096495856909314e-06, 2.6900317320149505e-06, 2.384185791015625e-05)

to get mean(abs(diff)) < 1e-5, in mlperf ref. implementation you need to do this:
  - in clip resblocks: self.mlp[1].approximate="tanh", to match tinygrad gelu fastmath
  - torch.backends.cuda.matmul.allow_tf32 = False
  - torch.backends.cudnn.allow_tf32 = False
  - torch.set_float32_matmul_precision("highest")
"""

if False:

  ### unet forward pass (during training), closeness testing

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
  # the actual training weights are zeroed out to start
  # in order to not get output of all zeroes, the unet_weights were set to nonzero values with: v.uniform_(-0.05, 0.05)
  unet_weights = safe_load("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/unet_training_init_model.safetensors")
  unet_io = safe_load("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/unet_training_io.safetensors")
  for k,v in unet_io.items():
    unet_io[k] = v.to(Device.DEFAULT).realize()

  load_state_dict(model, unet_weights)

  out = model(unet_io['x'], unet_io['timesteps'], unet_io['context']).realize()

  diff, ratio = md(unet_io['out'], out)

  end = 1