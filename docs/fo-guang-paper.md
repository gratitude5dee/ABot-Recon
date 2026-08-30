# Fo Guang — Buddha's Light: Turning the Surveillance State into a Public Good as a Resource for Rapid Response

*A position and systems paper building on ABot-Recon*

## Abstract

Cities have already paid for the sensors. Hundreds of millions of fixed CCTV cameras, traffic cameras, transit cameras, doorbell cameras, and body-worn cameras record the built environment continuously — yet in the minutes after an earthquake, flood, structural collapse, or wildfire, responders still operate on paper maps, phone calls, and whatever a drone can capture once one arrives. We argue that the marginal capability gap between "surveillance state" and "civic rapid-response infrastructure" is no longer sensing; it is *geometry at the edge*. Fo Guang ("Buddha's Light") is a proposal and reference design that repurposes existing camera streams into live, metrically consistent 3D reconstructions for emergency response, built on ABot-Recon, a streaming 3D reconstruction model with a fixed 12-frame local context window that produces camera trajectories, dense point maps, and confidence estimates from ordinary monocular video.

The pivotal enabler — and the core technical contribution we highlight — is ABot-Recon's **CPU inference path**. Surveillance infrastructure is overwhelmingly GPU-free: DVRs, NVRs, municipal edge boxes, and donated commodity hardware. We describe the engineering that makes a 1.0-billion-parameter geometry transformer run end-to-end on such hardware: a numerically bit-identical pure-PyTorch 2D rotary position encoding replacing a CUDA-only kernel, device-aware attention-backend resolution with a paged-KV cache that degrades gracefully from FlashInfer to native SDPA, device-agnostic autocast, and a *selectively scoped* dynamic INT8 quantization path whose per-submodule accuracy ablation shows why naive whole-model quantization destroys geometric output (~100% relative L2 on point maps) while a measured scope costs under 1%. On 8 commodity x86 cores the full pipeline runs at ~0.31 FPS — roughly 80× slower than the 24.45 FPS H100 baseline, yet fast enough for the after-action and slow-scan regimes that dominate civil-protection use, and deployable on hardware that already sits in every camera closet. We close with a governance design — dual-key activation, geometry-only egress, and public audit — under which surveillance optics become a public good without becoming a deeper surveillance state.

## 1. Introduction

### 1.1 The stranded asset

The world's installed camera base is the largest distributed sensor network ever built, and it is almost entirely *stranded* as a civic asset. Its output is consumed by three narrow pipelines: live human monitoring, forensic playback after crimes, and increasingly, person-centric analytics (face recognition, re-identification, behavior flagging) — the pipeline that earns the label "surveillance state." None of these pipelines produce the artifact that rapid response actually needs: an up-to-date, metrically consistent 3D model of the environment — which streets are passable, which façades have shifted, where debris fields begin, how floodwater is propagating.

That artifact used to require survey crews, LiDAR trucks, or photogrammetry flights measured in days. The recent generation of feed-forward geometry transformers — DUSt3R-lineage models such as Pi3, and streaming systems such as ABot-Recon built upon them — collapse it to a forward pass: monocular video in, camera trajectory and dense point maps out, no calibration targets, no SfM pipeline, no depth sensor.

### 1.2 Why the bottleneck is CPU

There is a hidden assumption in nearly all of this literature: an NVIDIA GPU is available. ABot-Recon's published efficiency figure — 24.45 FPS at 504×280 on an H100 — is impressive precisely because the H100 is a $30,000 data-center part. Municipal camera infrastructure contains no H100s. It contains:

- NVRs and DVRs built around low-power x86 or ARM SoCs;
- traffic-management cabinets with fanless industrial PCs;
- emergency-operations-center workstations bought on 5-year refresh cycles;
- in the developing world and in disaster zones, *whatever laptop survived*.

A civic rapid-response capability that requires shipping GPUs into a disaster area is not a capability; it is a procurement program. The deciding question for Fo Guang is therefore not "how fast is the model on an H100?" but "does it run *at all*, correctly, on the machines that already exist next to the cameras?" Until recently the answer for ABot-Recon — as for most of its peers — was no: the model could not even be *imported* without a compiled CUDA extension.

### 1.3 Contributions

This paper makes three contributions:

1. **A systems account of CPU enablement** (§3): the concrete engineering that takes a CUDA-native streaming geometry transformer to a correct, tested, end-to-end CPU path — RoPE kernel replacement with bit-identical semantics, attention-backend resolution, paged-KV fallback, and device-agnostic mixed-precision handling — with the design rules we extracted for other models in this class.
2. **A quantization ablation for geometry models** (§4): per-submodule dynamic INT8 measurements against an FP32 baseline showing a sharp cliff — quantizing the point decoder or the shared decoder blocks moves local point maps by ~100% relative L2 (i.e., destroys them), while a selective scope (camera decoder + prediction heads) costs 2.1 mm mean ATE, 0.68° mean rotation, and <1% on point and confidence maps. We argue that *scope selection by measured geometric regression*, not layer count, is the correct quantization methodology for reconstruction models.
3. **The Fo Guang architecture and governance design** (§5–§6): a tiered deployment model mapping reconstruction regimes (live, slow-scan, after-action) onto the hardware actually present in camera networks, and a dual-key, geometry-only-egress governance framework under which the capability is a public good rather than a surveillance deepening.

## 2. Background: ABot-Recon

ABot-Recon is a streaming 3D reconstruction model derived from the Pi3 architecture. A DINOv2 ViT-L encoder embeds each incoming frame; 36 shared transformer decoder blocks with 2D rotary position encoding attend over a **fixed 12-frame local context window** held in a KV cache; task-specific decoders and heads emit, per frame:

- a dense **local point map** (per-pixel 3D in camera coordinates, 280×504),
- a **relative pose** T_{i-1←i} to the previous frame,
- a per-pixel **confidence map** in [0,1].

Global trajectories and world point clouds are obtained by composing relative poses; a motion-visual rotation refiner suppresses drift, and an optional loop-closure stage (DINOv2-SALAD retrieval + sparse pose-graph optimization) refines long sequences with revisits. Because the context window is fixed, memory and compute are constant in stream length — the property that makes unbounded, always-on civic operation thinkable at all.

Total parameters: ~1.0B. Released checkpoint: ~4.0 GB of FP32 SafeTensors. Reference throughput: 24.45 FPS / 6.71 GiB at 504×280 on an H100, using FlashInfer paged-KV attention and a compiled cuRoPE CUDA kernel.

## 3. The CPU Optimization Innovation

The CPU path is not a port; it is a set of *dispatch corrections* that let one codebase serve both regimes. We describe each blocker, its fix, and the general rule.

### 3.1 Rotary position encoding: bit-identical PyTorch fallback

**Blocker.** The 2D RoPE module imported its compiled CUDA extension unconditionally; on any machine without the built kernel, the package failed at import time. The GPU-optional feature was, in practice, GPU-mandatory.

**Fix.** The extension import becomes optional (`_kernels = None` on failure), and the forward pass dispatches per call:

```python
if not (tokens.is_cuda and kernels_available()):
    return rope_2d_torch(tokens, positions, self.base, self.F0)
```

The subtlety is *numerical identity*. A generic PyTorch RoPE would be merely "close" to the kernel; geometric models compound pose over thousands of frames, so systematic sub-ulp biases become trajectory drift. `rope_2d_torch` reproduces two kernel-specific behaviors exactly: (i) the frequency-scaling factor `F0` is folded into the inverse frequencies (so `F0=-1` performs the inverse rotation, used by the paged-KV path to un-rotate cached keys), and (ii) every intermediate is rounded to the token dtype at the same points the kernel rounds. The result is `torch.equal`-exact against the repository's parity reference for fp32, fp16, and bf16, verified by CPU-side unit tests, and the round trip `RoPE(F0=-1)(RoPE(x))` recovers the input to 2×10⁻⁷.

**Rule.** *A fallback for a geometry model must match the kernel's rounding schedule, not just its formula.*

### 3.2 Attention: device-aware backend resolution and paged-KV degradation

**Blocker.** Backend selection consulted only FlashInfer availability, and the paged KV-cache manager refused to construct without FlashInfer — but FlashInfer's kernels are CUDA-only regardless of whether the wheel installs.

**Fix.** Resolution takes the device: on non-CUDA devices, `auto` resolves to PyTorch SDPA and an *explicit* `paged` request fails loudly (silent fallback would misreport benchmarked configurations). Independently, `PagedKVCacheManager` now degrades internally: when FlashInfer is unusable it keeps all of its paging semantics — one page per frame, ring-buffer window eviction, reference/summary pools — and swaps only the attention arithmetic for a gather of visible pages plus `scaled_dot_product_attention` in FP32. Unit tests verify the CPU manager's output equals dense SDPA over the visible window and that page recycling keeps the pool constant across an 8-frame stream with a 3-frame window.

This separation matters operationally: the *bookkeeping* (what the model may attend to — the fixed civic-memory window) is policy; the *kernel* (how fast attention runs) is capability. Fo Guang wants the policy identical on every tier of hardware and the kernel free to vary.

**Rule.** *Separate cache policy from attention kernel; make the policy device-invariant and let only the kernel dispatch on device.*

### 3.3 Device-agnostic mixed precision

**Blocker.** Five code sites hardcoded `torch.amp.autocast(device_type="cuda", ...)` — in the streaming frame-forward paths, the Pi3 forward, and the camera head — and `torch.cuda.empty_cache()` ran unconditionally after checkpoint load. These are the classic "works on my A100" assumptions.

**Fix.** Autocast device types derive from the tensors at hand (`hidden.device.type`, `feat.device.type`); CUDA cache maintenance is guarded by `torch.cuda.is_available()`. At the wrapper level, autocast is enabled only when the device is CUDA *and* the compute dtype is not FP32, so CPU inference runs in clean FP32 without a no-op autocast context; input frames are materialized in FP32 on CPU rather than bf16.

**Rule.** *Derive device types from data, never from configuration defaults; precision policy is a function of (device, dtype), not a constant.*

### 3.4 Result: end-to-end CPU inference

With these three families of fixes, `ABotRecon.from_pretrained(checkpoint, device="cpu")` loads the released 4 GB checkpoint and streams video end-to-end with no CUDA runtime, no FlashInfer, and no compiled extension. Measured on 12 real frames (TUM freiburg1_desk) at 504×280 on 8 cores of an Intel Xeon Platinum 8559C:

| Configuration | Throughput | Peak RSS |
|---|---|---|
| H100 (paper baseline) | 24.45 FPS | 6.71 GiB |
| CPU FP32, SDPA, PyTorch RoPE | ~0.31 FPS (~3.2 s/frame) | ~10 GiB |
| CPU + selective INT8 (§4) | ~0.32 FPS | ~10 GiB peak; ~0.4 GiB less resident weights |

We state the honest number prominently: **the CPU is ~80× slower than the H100.** Section 5 shows why this is nevertheless the number that matters.

## 4. Selective Dynamic INT8: Quantization with a Geometric Conscience

Dynamic INT8 quantization (`torch.ao.quantization.quantize_dynamic` on `nn.Linear`) is the standard low-effort CPU optimization, and for classification or language models it is usually benign. For a geometry-regression model it is not, and the failure is invisible unless you measure the *geometry*, not the loss.

We ablated quantization scope submodule-by-submodule against the FP32 CPU baseline on the same 12-frame sequence, reporting trajectory error (ATE against the FP32 trajectory, whose total length is 63 mm), rotation deviation, and relative L2 on dense outputs:

| Quantized scope | ATE mean | Rotation mean | Point maps rel. L2 | Confidence rel. L2 |
|---|---|---|---|---|
| camera decoder + all heads *(shipped)* | 2.1 mm | 0.68° | **0.6%** | **0.8%** |
| + confidence decoder | 2.1 mm | 0.68° | 0.6% | **51%** |
| + point decoder | 2.1 mm | 0.68° | **105%** | 51% |
| + shared decoder blocks (36) | 9.5 mm | 0.91° | 106% | 51% |

Three observations:

1. **The cliff is per-submodule, not gradual.** Quantizing the point decoder alone moves the dense point maps by ~100% relative L2 — the reconstruction is destroyed — while leaving the *trajectory* untouched (the camera path flows through a different decoder). A pipeline that validated only ATE would ship a model that navigates correctly through geometry that no longer exists.
2. **Sensitivity follows output dynamic range, not parameter count.** The 36 shared decoder blocks (the bulk of parameters and compute) degrade the trajectory modestly; the small point decoder is catastrophic. Per-token activation ranges in the point pathway span the scene's full metric depth, and dynamic per-tensor INT8 cannot represent that range at the precision dense regression requires.
3. **The honest benefit is memory, not speed.** With the shipped scope, throughput is unchanged within noise (~0.31→0.32 FPS) because CPU time is dominated by the FP32 DINOv2 encoder and shared decoder blocks — exactly the modules that must stay FP32. The win is ~0.4 GiB of resident weight memory, which matters on 8 GB edge boxes running alongside an NVR process.

The shipped configuration therefore quantizes only the camera decoder, the global-points decoder, and all prediction heads; it is exposed as `--quantize` in the CLI and `quantization="int8_dynamic"` in the API, is rejected on non-CPU devices, and is recorded in inference metadata so downstream consumers (and auditors — §6) can see exactly which numeric regime produced a given reconstruction.

**Methodological claim.** For reconstruction models, quantization scope must be selected by *measured geometric regression per submodule* — ATE, rotation, and dense-output relative error jointly — never by global proxy metrics. We offer the table above as a template.

## 5. Fo Guang: Architecture for Rapid Response

### 5.1 Three response regimes

Rapid response does not need 24 FPS everywhere. It needs the right latency in the right place:

- **Live regime** (command posts, drone feeds): seconds-fresh geometry over a handful of priority streams. Served by whatever GPUs the jurisdiction has — a single workstation GPU sustains multiple streams at the H100-class model's efficiency point.
- **Slow-scan regime** (fixed camera canvassing): each of hundreds of fixed cameras contributes a frame every few seconds; scene change, not motion, is the signal. At ~3.2 s/frame, *one 8-core CPU box saturates one camera's slow scan* — and every NVR already contains such a box. The CPU path makes the entire installed camera base addressable without new hardware.
- **After-action regime** (damage assessment, insurance, forensics of the disaster itself): overnight batch reconstruction of the day's footage. Throughput is irrelevant; correctness and auditability are everything. Runs on donated laptops, library computers, cloud spot CPU — anything.

The CPU optimization is what makes tiers two and three exist. Without it, Fo Guang is a proposal to buy GPUs; with it, Fo Guang is a software update to hardware already deployed.

### 5.2 Reference pipeline

```
fixed cameras / body cams / drones / robots (e.g. Unitree G1 head camera)
        │  JPEG/RTSP frames, lexicographically ordered
        ▼
ABot-Recon streaming inference  (device=cuda | cpu, --quantize on CPU edge)
        │  per frame: local points, relative pose, confidence
        ▼
pose composition + optional loop closure (SALAD retrieval + sparse PGO)
        │  world points, refined trajectory
        ▼
confidence-thresholded PLY / rolling-window streamer
        ▼
GEV (God's-Eye-View) browser viewer — the common operating picture
```

Every stage above exists in the ABot-Recon repository today (`demo.py`, `scripts/stream_reconstruction.py`, `scripts/export_reconstruction_ply.py`, the GEV consumer); Fo Guang's contribution at this layer is the *deployment mapping* — which regime runs where — plus the governance boundary of §6. The fixed 12-frame context window gives every tier identical, bounded memory behavior: a camera that streams for a month costs the same RAM as one that streams for a minute.

### 5.3 Why monocular reconstruction is the right primitive

Rapid response has tried and abandoned heavier primitives. LiDAR maps are precise but stale and sparse in coverage; photogrammetry requires flight operations; SLAM stacks require calibration and per-deployment tuning. ABot-Recon's primitive — *uncalibrated monocular video to metric-consistent geometry with per-pixel confidence* — matches the actual inventory: cameras of unknown intrinsics, mounted years ago, by many vendors. The confidence map is operationally essential: responders must know which parts of the model to trust before committing entry teams, and `--confidence-threshold` gives incident commanders a single dial from "show everything" to "show only what the model is sure of."

## 6. Governance: Light, Not Gaze

A system that ingests every public camera is a hazard as much as an asset. Fo Guang's governance design rests on one technical fact and three policy mechanisms.

**The technical fact: geometry is not identity.** The Fo Guang egress is point clouds, trajectories, and confidence maps at 280×504. This output class does not carry faces, gait signatures, license plates, or re-identifiable appearance; the reconstruction of a street tells you the street collapsed, not who was standing on it. By fixing the *output type* at the architecture level — the model has no identity heads to enable — the privacy property is structural rather than promised.

**Mechanism 1: dual-key activation.** Camera streams are not connected to Fo Guang by default. Ingest requires a declared emergency (executive key) *and* an independent authorization (judicial or ombuds key), scoped in geography and time, after which access lapses automatically. The default state is off.

**Mechanism 2: geometry-only egress at the edge.** Reconstruction runs *on the NVR/edge box beside the camera* — feasible precisely because of the CPU path — and only geometry leaves the premises. Raw pixels never traverse the network to a central authority. The CPU optimization is thus not merely a cost story; it is the *privacy architecture*: computation moves to the data so that imagery does not move to the state.

**Mechanism 3: public audit of the pipeline itself.** Inference metadata records device, attention backend, quantization mode, and confidence threshold per run; checkpoints are content-addressed (SafeTensors digests); the model, tests, and this paper's ablation are public. Anyone can verify that the deployed artifact computes geometry and nothing else, and reproduce its accuracy envelope on a laptop — because it runs on a laptop.

We name the project for this inversion. Fo Guang (佛光), "Buddha's light," is illumination that falls on everyone and singles out no one. The same optics that constitute a surveillance state when pointed at *persons* constitute a public good when constrained — technically and institutionally — to *places*.

## 7. Limitations and Future Work

- **Throughput honesty.** 0.31 FPS on 8 cores serves slow-scan and after-action regimes, not live video. Closing the gap on CPU likely requires INT8 or bf16 kernels for the encoder and shared decoder blocks with quantization-aware finetuning — our ablation shows post-training dynamic INT8 cannot reach them safely.
- **Single-sequence evaluation.** The accuracy deltas of §4 are measured on one 12-frame indoor sequence; a civic deployment demands the ablation re-run across the model's evaluation suites (KITTI, TUM, ETH3D-class data) and across seasons/weather for fixed cameras.
- **Multi-camera fusion.** Fo Guang currently reconstructs per-stream and fuses in the viewer; principled cross-camera registration (shared loop-closure graph across the network) is the natural next system.
- **Adversarial governance.** Dual-key regimes fail when both keys are held by aligned actors; the audit mechanism deters but does not prevent scope creep. We consider the structural output-type constraint the strongest of the three mechanisms and the one deserving formalization (e.g., attested edge binaries).

## 8. Conclusion

The surveillance state's cameras are a sunk cost; its harms come from what is computed on them. ABot-Recon's CPU path — a bit-identical RoPE fallback, device-aware attention with a gracefully degrading paged-KV cache, device-agnostic precision handling, and a geometrically-audited selective INT8 mode — demonstrates that billion-parameter reconstruction now runs on the commodity processors already wired to those cameras. That single systems fact reorders the possibility space: geometry can be computed where the pixels live, identity need never leave the building, and the output — a live map of a stricken city — belongs to everyone it illuminates. Turning the surveillance state into a public good is not a metaphor awaiting technology. As of the CPU path, it is a deployment decision.

## References

1. ABot-Recon: Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction. Model release, checkpoint `acvlab/ABot-Recon` (Hugging Face). Repository README and technical report.
2. Wang, Y. et al. Pi3: Permutation-equivariant visual geometry learning. (Upstream architecture of ABot-Recon.)
3. Wang, S. et al. DUSt3R: Geometric 3D vision made easy. CVPR 2024.
4. Oquab, M. et al. DINOv2: Learning robust visual features without supervision. TMLR 2024.
5. Izquierdo, S. and Civera, J. Optimal transport aggregation for visual place recognition (SALAD). CVPR 2024.
6. Ye, Z. et al. FlashInfer: Efficient and customizable attention engine for LLM inference serving. MLSys 2025.
7. Su, J. et al. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing 2024.
8. PyTorch Team. `torch.ao.quantization` — dynamic quantization. PyTorch 2.5 documentation.
9. Sturm, J. et al. A benchmark for the evaluation of RGB-D SLAM systems (TUM RGB-D). IROS 2012.
10. This repository: CPU inference and selective INT8 path, PR #4 (`gratitude5dee/ABot-Recon`), including the per-submodule quantization ablation reproduced in §4.
