from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from typing import Any
from collections import namedtuple
from examples.stable_diffusion import get_alphas_cumprod
from pathlib import Path
BASEDIR = Path("/home/hooved/train-sd/training/stable_diffusion")

# goal: translate mlperf stable_diffusion-v2 ref. implementation to tinygrad

# not shown: generation of safetensors from mlperf reference implementation
# to get these safetensors, setup the mlperf docker image and config (see README), and export the safetensors where indicated in the commented mlperf code

compare_weights_three_steps = True
compare_weights_onestep = False
compare_grads = False
compare_end_to_end = False
compare_latent = False
compare_clip = False
compare_unet = False
compare_loss = False
"""
for all comparisons, used these settings which are different than default mlperf settings, in order to enable apples-to-apples comparison with tinygrad math
- torch.backends.cuda.matmul.allow_tf32 = False
- torch.backends.cudnn.allow_tf32 = False
- torch.set_float32_matmul_precision("highest")
- in configs/train_01x01x01.yaml:
  - change model.params.unet_config.use_fp16 from true to false
  - comment out this setting (so math is in fp32): lightning.trainer.precision: 16

for compare_clip only:
- in clip resblocks: self.mlp[1].approximate="tanh", to match tinygrad gelu fastmath
"""

def md(a, b):
  diff = (a - b).abs()
  max_diff = diff.max()
  mean_diff = diff.mean()
  ratio = mean_diff / a.abs().mean()
  return mean_diff.item(), ratio.item(), max_diff.item()

alphas_cumprod = get_alphas_cumprod()
sqrt_alphas_cumprod = alphas_cumprod.sqrt()
sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

if compare_weights_three_steps:
  DATADIR = BASEDIR / "checkpoints"
  tiny = safe_load(DATADIR / "tiny_out.2.bias_after_3_steps.safetensors")
  ref = safe_load(DATADIR / "ref_out.2.bias_after_3_training_steps.safetensors")
  """
  tiny["out.2.bias"].tolist()
  # [-1.248389863706123e-10, -1.2538842186771149e-10, -8.42087788388568e-11, 1.2096189327959195e-10]
  ref["model.diffusion_model.out.2.bias"].tolist()
  # [-3.7063471736153986e-10, -3.4958785866123776e-10, -1.9371104720278254e-10, 3.665407977138102e-10]
  """
  md(ref["model.diffusion_model.out.2.bias"].to("NV"), tiny["out.2.bias"].to("NV"))
  #(2.0626908514564946e-10, 0.6443520784378052, 2.4579571711313974e-10)

  pause = 1

if compare_weights_onestep:
  DATADIR = BASEDIR / "checkpoints"
  tiny = safe_load(DATADIR / "tiny_out.2.bias_after_1_steps.safetensors")
  ref = safe_load(DATADIR / "model_after_1_training_steps.safetensors")
  """
  tiny["out.2.bias"].tolist()
  [-1.2499917279636813e-13, -1.2499915924384097e-13, -1.2499914569131382e-13, 1.2499917279636813e-13]
  ref["model.diffusion_model.out.2.bias"].tolist()
  [-1.249999859479975e-13, -1.2499997239547034e-13, -1.2499995884294318e-13, 1.2499999950052465e-13]
  """
  md(ref["model.diffusion_model.out.2.bias"].to("NV"), tiny["out.2.bias"].to("NV"))
  # (8.165397611531455e-19, 6.532319275720511e-06, 8.267041565201971e-19)

  pause = 1

if compare_grads:
  DATADIR = BASEDIR / "checkpoints"

  #init_weights = safe_load(DATADIR / "training_init_model.safetensors")
  #ref_grads = safe_load(DATADIR / "grads_after_1_training_steps.safetensors")
  #tiny_grads_backup = safe_load(DATADIR / "tiny_grads_after_1_steps.backup.safetensors")

  tiny_grads = safe_load(DATADIR / "tiny_grads_after_1_steps.safetensors")
  ref_grads = safe_load(DATADIR / "out.2.weight_after_1_training_steps.safetensors")
  md(ref_grads["model.diffusion_model.out.2.bias.grad"].to("NV"), tiny_grads["out.2.bias.grad"].to("NV"))
  #(3.1656527426093817e-06, 8.865074050845578e-05, 8.277595043182373e-06)
  #(6.505288183689117e-05, 0.0001320305309491232, 0.00010508298873901367)

  pause = 1

if compare_end_to_end:
  DATADIR = BASEDIR / "checkpoints"

  ref_losses = safe_load(DATADIR / "eleven_training_steps.safetensors")["loss"]
  ref_losses.to_("CPU").realize()

  tiny_losses = safe_load(DATADIR / "tiny_losses.safetensors")
  tiny_losses = Tensor.stack(*[tiny_losses[str(i)].to_("CPU") for i in range(len(tiny_losses))]).unsqueeze(1).realize()
  mean_diff, ratio, max_diff = md(ref_losses, tiny_losses)
  #(0.00018372319755144417, 0.00015886307915206042, 0.00032806396484375)

  #### measure mean(abs(diff)) for parameter values
  tiny_weights = safe_load(DATADIR / "tiny_after_eleven_training_steps.safetensors")
  #for v in tiny_weights.values(): v.to_("NV")

  tiny_keys = sorted(tiny_weights.keys())

  ref_weights = safe_load(DATADIR / "model_after_eleven_training_steps.safetensors")
  #for v in ref_weights.values(): v.to_("NV")

  # 866M params * 2 (plus bufs) is too much for 10GB GPU memory
  #all_tiny = Tensor.cat(*[tiny_weights[k].flatten() for k in tiny_keys])
  #all_ref = Tensor.cat(*[ref_weights["model.diffusion_model." + k].flatten() for k in tiny_keys])
  #mean_diff, ratio, max_diff = md(all_ref, all_tiny)

  total_diff_sum = Tensor(0.0)
  total_a_abs_sum = Tensor(0.0)
  total_count = 0
  global_max_diff = Tensor([0.0])

  for k in tiny_keys:
    a = ref_weights["model.diffusion_model." + k].to_("NV")
    b = tiny_weights[k].to_("NV")
    diff = (a - b).abs()

    total_diff_sum   += diff.sum().realize()
    total_a_abs_sum  += a.abs().sum().realize()
    total_count      += diff.numel()
    #global_max_diff = max(global_max_diff, diff.max().item())
    global_max_diff = global_max_diff.cat(diff.max().unsqueeze(0)).realize()

  global_max_diff = global_max_diff.max().item()
  mean_diff = (total_diff_sum / total_count).item()
  ratio     = (total_diff_sum / total_a_abs_sum).item()
  print("all params diff")
  print((mean_diff, ratio, global_max_diff))
  #(1.5695044005767378e-12, 2.3212920474691146e-10, 5.624485321931161e-09)
  
  pause = 1

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

  ### add noise (for training input)
  # data['t'].shape == (1,)
  # data['t'].dtype == dtypes.long
  # noise = Tensor.randn_like(latent)
  latent_with_noise = sqrt_alphas_cumprod[data['t']] * latent + sqrt_one_minus_alphas_cumprod[data['t']] * data['noise']
  mean_diff, ratio, max_diff = md(data['x_noisy'], latent_with_noise)
  # (2.717769348237198e-05, 3.34946510299305e-05, 0.0005095005035400391)

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

if compare_loss:
  data = safe_load("/home/hooved/train-sd/training/stable_diffusion/datasets/tensors/unet_inputs.safetensors")
  for k,v in data.items():
    data[k] = v.to("NV").realize()
  #out = model(data['x_noisy'], data['t'], data['cond']).realize()
  v_actual = sqrt_alphas_cumprod[data['t']] * data['noise'] - sqrt_one_minus_alphas_cumprod[data['t']] * data['x']
  loss = ((data['model_output'] - v_actual) ** 2).mean() / v_actual.shape[0]
  mean_diff, ratio, max_diff = md(data['loss'], loss)
  # (2.384185791015625e-07, 1.9723628571727243e-07, 2.384185791015625e-07)

  end = 1

"""
  def md(a, b):
    diff = (a - b).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    ratio = mean_diff / a.abs().mean()
    return mean_diff.item(), ratio.item(), max_diff.item()
  data = safe_load(BASEDIR / "checkpoints" / "linear.safetensors")
  for w in data.values():
    w.to_("NV").realize()
  from tinygrad.nn import Linear
  layer = Linear(*data['self.to_q.weight'].shape, bias=False)
  layer.weight = data['self.to_q.weight'].cast(dtypes.float16)
  tiny_q = layer(data['x'].cast(dtypes.float16))
  print(md(data['q'], tiny_q))
  pause = 1
"""