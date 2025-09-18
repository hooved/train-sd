import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

clip_series, fid_series = {}, {}

# one of the reference runs at BS=512, lr=1.25e-7 * 512 = 6.4e-5
steps = list(range(1000, 11000, 1000))
M_images = [s * 512 / 1_000_000 for s in steps]
clip = [0.07037353515625, 0.095458984375, 0.1297607421875, 0.1490478515625, 0.1700439453125, 0.175537109375, 0.1998291015625, 0.19580078125,
        0.1925048828125, 0.2086181640625]
fid = [240.307753187222, 160.470078270219, 116.104976258075, 98.6194527639151, 77.9144665349304, 78.277784436582, 61.9517434630711, 57.6910232633396,
       64.6372463854434, 59.762565549819]
clip_series.update({"(mlperf ref) BS=512, f16, lr=6.4e-5": (np.array(M_images), np.array(clip))})
fid_series.update({"(mlperf ref) BS=512, f16, lr=6.4e-5": (np.array(M_images), np.array(fid))})

"""
# BS=336, bf16, all of softmax in fp32, lr=1.25e-7
steps = [762, 6096]
M_images = [s * 336 / 1_000_000 for s in steps]
clip = [0.038616277277469635, 0.0429423525929451]
fid = [420.8504498242364, 403.2341038707751]
clip_series.update({"BS=336, bf16, lr=1.25e-7": (np.array(M_images), np.array(clip))})
fid_series.update({"BS=336, bf16, lr=1.25e-7": (np.array(M_images), np.array(fid))})

# BS=512 (gradacc), f32, lr=1.25e-7
steps = [500, 1000, 5000, 6000]
M_images = [s * 512 / 1_000_000 for s in steps]
clip = [0.04045993834733963, 0.034708019345998764, 0.05408899113535881, 0.05153243616223335]
fid = [425.648211214712, 409.4558078361266, 375.82205725656814, 385.55911118460483]
clip_series.update({"BS=512, f32, lr=1.25e-7": (np.array(M_images), np.array(clip))})
fid_series.update({"BS=512, f32, lr=1.25e-7": (np.array(M_images), np.array(fid))})
"""

# BS=512 (gradacc), f32, lr=1.25e-7 * 512 = 6.4e-5
steps = [1000, 3000, 6000]
M_images = [s * 512 / 1_000_000 for s in steps]
clip = [0.07314359396696091, 0.10343340784311295, 0.11351927369832993]
fid = [245.59511952673307, 150.90382147424623, 136.49630935835017]
clip_series.update({"BS=512, f32, lr=6.4e-5": (np.array(M_images), np.array(clip))})
fid_series.update({"BS=512, f32, lr=6.4e-5": (np.array(M_images), np.array(fid))})

# BS=336, bf16, all of softmax in fp32, lr=4.2e-5
# /home/hooved/stable_diffusion/checkpoints/training_checkpoints/09090228 to ckpt 5334
# /home/hooved/stable_diffusion/checkpoints/training_checkpoints/09100305 to ckpt 15240
steps = [5334, 15240]
M_images = [s * 336 / 1_000_000 for s in steps]
clip = [0.13652677834033966, 0.16824787855148315]
fid = [115.780179659493, 80.58007408795413]
clip_series.update({"BS=336, bf16, lr=4.2e-5": (np.array(M_images), np.array(clip))})
fid_series.update({"BS=336, bf16, lr=4.2e-5": (np.array(M_images), np.array(fid))})

# BS=304, bf16, fp32 softmax, lr=3.8e-5, recent rebase (try 1)
"""
steps = [11795]
M_images = [s * 304 / 1_000_000 for s in steps]
clip = [0.13711006939411163]
fid = [103.2172381597046]
clip_series.update({"BS=304, bf16, lr=3.8e-5 (try 1)": (np.array(M_images), np.array(clip))})
fid_series.update({"BS=304, bf16, lr=3.8e-5 (try 1)": (np.array(M_images), np.array(fid))})
"""

# BS=304, bf16, fp32 softmax, lr=3.8e-5, recent rebase (try 2)
# /home/hooved/stable_diffusion/checkpoints/training_checkpoints/09130207
steps = [1685, 3370, 5055, 8425, 10110, 13480, 15165, 16850]
M_images = [s * 304 / 1_000_000 for s in steps]
clip = [0.06879530847072601, 0.10878439247608185, 0.14284193515777588, 0.13588756322860718, 0.15333493053913116, 0.13692690432071686, 0.16351738572120667, 0.15678671002388]
fid = [216.85881074283574, 141.156650531757, 118.97741769816344, 112.09800821460959, 96.10085130500028, 111.69978282075118, 92.64104949522562, 94.75978545539562]
clip_series.update({"BS=304, bf16, lr=3.8e-5": (np.array(M_images), np.array(clip))})
fid_series.update({"BS=304, bf16, lr=3.8e-5": (np.array(M_images), np.array(fid))})

# BS=304, bf16, fp32 softmax, lr=1.9e-5, recent rebase
# /home/hooved/stable_diffusion/checkpoints/training_checkpoints/09141228
steps = [8425, 10110]
M_images = [s * 304 / 1_000_000 for s in steps]
clip = [0.113297238945961, 0.10917766392230988]
fid = [137.18308790806293, 142.33904917579667]
clip_series.update({"BS=304, bf16, lr=1.9e-5": (np.array(M_images), np.array(clip))})
fid_series.update({"BS=304, bf16, lr=1.9e-5": (np.array(M_images), np.array(fid))})


# BS=304, bf16, fp32 softmax, lr=5.7e-5, recent rebase
# /home/hooved/stable_diffusion/checkpoints/training_checkpoints/09150331 to ckpt 6740
# /home/hooved/stable_diffusion/checkpoints/training_checkpoints/09151858 from 09150331/6740 to ckpt 10110
steps = [8425, 10110]
M_images = [s * 304 / 1_000_000 for s in steps]
clip = [0.16371633112430573, 0.16669005155563354]
fid = [101.22468192693611, 93.23531995164893]
clip_series.update({"BS=304, bf16, lr=5.7e-5": (np.array(M_images), np.array(clip))})
fid_series.update({"BS=304, bf16, lr=5.7e-5": (np.array(M_images), np.array(fid))})

# BS=304, bf16, fp32 softmax, lr=7.6e-5, recent rebase
# /home/hooved/stable_diffusion/checkpoints/training_checkpoints/09160330
steps = [8425, 10110]
M_images = [s * 304 / 1_000_000 for s in steps]
clip = [0.15790054202079773, 0.16030366718769073]
fid = [97.1283841862379, 85.26515191821738]
clip_series.update({"BS=304, bf16, lr=7.6e-5 (i=1)": (np.array(M_images), np.array(clip))})
fid_series.update({"BS=304, bf16, lr=7.6e-5 (i=1)": (np.array(M_images), np.array(fid))})

"""
16-Sep-2025
BS=304, bf16, fp32 softmax, lr=7.6e-5, recent rebase
I want to show these hyparams are replicable for converging before 5.1M images (ideally before 3.1M images).
Running duplicate runs so I can get N=3

tinyamd1:
/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09162104
crashed with `Bus error` after backup_3370.safetensors

tinyamd2:
/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09162050
manually stopped after backup_5055.safetensors due to power downtime
"""

# 17-Sep-2025
# BS=304, bf16, fp32 softmax, lr=7.6e-5, recent rebase
# /home/hooved/stable_diffusion/checkpoints/training_checkpoints/09172111
steps = [8425, 10110]
M_images = [s * 304 / 1_000_000 for s in steps]
clip = [0.17256876826286316, 0.17909857630729675]
fid = [79.61212198096996, 77.126169084374]
clip_series.update({"BS=304, bf16, lr=7.6e-5 (i=2)": (np.array(M_images), np.array(clip))})
fid_series.update({"BS=304, bf16, lr=7.6e-5 (i=2)": (np.array(M_images), np.array(fid))})


### graphing

series_list = [clip_series, fid_series]
y_labels    = ["clip score", "fid score"]
titles      = ["clip eval", "fid eval"]

# Consistent colors per label across both panels
all_labels = sorted({lab for s in series_list for lab in s.keys()})
palette = list(plt.cm.tab10.colors)

custom_colors = {
    "(mlperf ref) BS=512, f16, lr=6.4e-5": "tab:blue",
    #"BS=400, bf16": "tab:green",
    #"BS=248, f16": "tab:orange",
    #"BS=336, bf16, lr=1.25e-7": "tab:red",
    #"BS=512, f32, lr=1.25e-7": "tab:purple",
    "BS=512, f32, lr=6.4e-5": "tab:green",
    "BS=336, bf16, lr=4.2e-5": "black",
    "BS=304, bf16, lr=3.8e-5": "crimson",
    "BS=304, bf16, lr=1.9e-5": "#ebc934",
    "BS=304, bf16, lr=5.7e-5": "#2bcfab",
    "BS=304, bf16, lr=7.6e-5 (i=1)": "#9e34eb",
    "BS=304, bf16, lr=7.6e-5 (i=2)": "#eb34b4",
}
#color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(all_labels)}
color_map = {lab: custom_colors.get(lab, palette[i % len(palette)]) for i, lab in enumerate(all_labels)}
custom_markers = {
    "BS=304, bf16, lr=3.8e-5": "^",
    "BS=304, bf16, lr=1.9e-5": "^",
    "BS=304, bf16, lr=5.7e-5": "^",
    "BS=304, bf16, lr=7.6e-5 (i=1)": "*",
    "BS=304, bf16, lr=7.6e-5 (i=2)": "*",
}

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

for ax, sdata, ylab, ttl in zip(axes, series_list, y_labels, titles):
    for lab, (x, y) in sdata.items():
        ax.scatter(x, y, s=25, alpha=0.85, label=lab, color=color_map[lab], **({"marker": custom_markers[lab]} if lab in custom_markers else {}))
        # NEW: connect points with a line (sorted by x)
        order = np.argsort(x)
        ax.plot(
            np.array(x)[order], np.array(y)[order],
            linewidth=1.2, alpha=0.85, color=color_map[lab]
            # no label here so the legend only shows one entry per series
        )
    ax.set_ylabel(ylab)
    ax.set_title(ttl)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    #ax.set_xlim(left=0)
    #ax.set_ylim(bottom=0)

axes[0].axhline(y=0.15, linestyle="--", color="red")
axes[1].axhline(y=90, linestyle="--", color="red")

axes[-1].set_xlabel("number of image samples (millions)")

# Single legend on the right
by_label = {}
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        by_label[l] = h

fig.tight_layout(rect=[0, 0, 0.85, 1])
fig.legend(
    by_label.values(),
    by_label.keys(),
    title="Series",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
)

outdir = Path("./")
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "clip_fid.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
