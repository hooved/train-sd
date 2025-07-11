from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from typing import Any
from collections import namedtuple

# goal: translate mlperf stable_diffusion-v2 ref. implementation to tinygrad

# not shown: generation of safetensors from mlperf reference implementation
# to get these safetensors, setup the mlperf docker image and config (see README), and export the safetensors where indicated in the commented mlperf code

compare_latent = False
compare_clip = False
compare_unet = True
"""
for all comparisons, used these settings which are different than default mlperf settings, in order to enable apples-to-apples comparison with tinygrad math
- torch.backends.cuda.matmul.allow_tf32 = False
- torch.backends.cudnn.allow_tf32 = False
- torch.set_float32_matmul_precision("highest")
- in configs/train_01x01x01.yaml:
  - change model.params.unet_config.use_fp16 from true to false
  - comment out this setting (so math is in fp32): lightning.trainer.precision: 16

for clip only:
- in clip resblocks: self.mlp[1].approximate="tanh", to match tinygrad gelu fastmath
"""

def md(a, b):
  diff = (a - b).abs()
  max_diff = diff.max().item()
  mean_diff = diff.mean().item()
  ratio = mean_diff / a.abs().mean().item()
  return mean_diff, ratio, max_diff

### sampled latent closeness testing
if compare_latent:
  data = safe_load("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/unet_inputs.safetensors")
  for k,v in data.items():
    data[k] = v.to("NV").realize()
  SCALE_FACTOR = 0.18215
  # sample latent from VAE-generated distribution (NOTE: mlperf ref. starts from mean/logvar loaded from disk, as done here)

  mean_logvar = data['batch'].cast(dtypes.half).squeeze(1)
  mean, logvar = Tensor.chunk(mean_logvar, 2, dim=1)
  std = Tensor.exp(0.5 * logvar.clamp(-30.0, 20.0))
  #latent = (mean + std * Tensor.randn(mean.shape)) * SCALE_FACTOR
  latent = (mean + std * data["latent_randn_sampling"]).cast(dtypes.float16) * SCALE_FACTOR
  mean_diff, ratio, max_diff = md(data['x'], latent)
  # (0.0001468658447265625, 0.0001968463678010471, 0.0026092529296875)

### clip encoding closeness testing
if compare_clip:
  """
  to get mean(abs(diff)) < 1e-5, in mlperf ref. implementation you need to do this:
    - in clip resblocks: self.mlp[1].approximate="tanh", to match tinygrad gelu fastmath
    - torch.backends.cuda.matmul.allow_tf32 = False
    - torch.backends.cudnn.allow_tf32 = False
    - torch.set_float32_matmul_precision("highest")
  """
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

  mean_diff, ratio, max_diff = md(data['c'], context)
  # (2.1096495856909314e-06, 2.6900317320149505e-06, 2.384185791015625e-05)

### unet forward pass (during training), closeness testing
if compare_unet:
  from extra.models.unet import UNetModel
  data = safe_load("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/unet_training_io.safetensors")
  for k,v in data.items():
    data[k] = v.to("NV").realize()

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
  load_state_dict(model, unet_weights)

  out = model(data['x'], data['timesteps'], data['context']).realize()

  mean_diff, ratio, max_diff = md(data['out'], out)
  # (9.668409006735601e-08, 2.3214517111372956e-06, 4.842877388000488e-07)

  end = 1