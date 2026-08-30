# Unitree G1 (Jetson Orin) capture → reconstruction runbook

End-to-end path from a G1 head-camera clip to a browser-ready point cloud.
Everything here is read-only perception: subscribe to an image topic, copy the
frames off, reconstruct elsewhere (or on the Orin), export. No joint is ever
commanded, developer mode is never entered, and network configuration is never
touched.

## 0. Safety preconditions

- Work in a disposable conda env named after yourself; remove it on exit.
- Keep the CCTV feed open for the whole session if the robot can move at all.
- Do not run a gait controller that has not been validated in simulation, and
  never without a tested damping shutdown (kp=0, kd≈8 on all 29 joints,
  trapping SIGINT/SIGTERM).
- Delete recordings and any checkpoint you staged on the robot before leaving.

## 1. Record a clip on the Orin

`scripts/record_g1_camera.py` needs only `unitree_sdk2py` and
`cyclonedds==0.10.2` — not this package — so it runs in a throwaway env:

```bash
ssh unitree@<orin-tailscale-ip>
conda create -n <yourname> python=3.10 -y && conda activate <yourname>
pip install cyclonedds==0.10.2 unitree_sdk2py

# Confirm the topic carries what the recorder expects before recording.
python record_g1_camera.py --output-dir /tmp/g1-clip --list-fields

python record_g1_camera.py --output-dir /tmp/g1-clip --fps 10 --max-seconds 30
```

The DDS topic and message class differ between SDK builds, so both are flags
(`--topic`, `--message-class`); the defaults match the SDK's front-video
sample. `--list-fields` prints the field names of the first received sample,
which is the fastest way to tell a wrong topic from a silent one. Ctrl-C stops
cleanly at any point.

Output is exactly what `demo.py --image-dir` wants — `000001.jpg`,
`000002.jpg`, ... in capture order — plus a `manifest.json` recording the
topic, requested rate, and the rate actually achieved.

Copy the frames off and clear the scratch dir:

```bash
rsync -a unitree@<orin>:/tmp/g1-clip/ ./g1-clip/
ssh unitree@<orin> 'rm -rf /tmp/g1-clip'
```

## 2a. Reconstruct off-board (CUDA 12.1 workstation)

The reliable path. Install per the [README](../README.md), place the checkpoint
at `checkpoints/abot_recon.safetensors`, then:

```bash
python demo.py \
  --image-dir g1-clip \
  --output-dir outputs/g1 \
  --attention-backend auto \
  --no-loop-closure \
  --save-world-points
```

## 2b. Reconstruct on the Orin

Possible, but the README install is x86-only: `pip install torch==2.5.1
--index-url .../whl/cu121` has no aarch64 wheel, and plain `pip install -e .`
resolves the `torch==` pin to the **CPU-only** PyPI `linux_aarch64` wheel,
quietly replacing a working JetPack CUDA build.

So install a CUDA-capable aarch64 torch first (JetPack wheel or NVIDIA's pip
index), then let the helper do the rest — it verifies `torch.cuda`, pins the
installed build (local `+nv...` segment included) in a constraints file so
transitive `torch` requirements resolve to it, and installs this package with
`--no-deps`:

```bash
python scripts/setup_jetson_env.py --dry-run   # show the pip plan
python scripts/setup_jetson_env.py
```

Inference then has to avoid the CUDA extras, neither of which is needed:
`--attention-backend sdpa` is a pure-PyTorch SDPA path (no FlashInfer), and
cuRoPE stays uncompiled.

```bash
python demo.py \
  --image-dir /tmp/g1-clip \
  --output-dir outputs/g1 \
  --attention-backend sdpa \
  --no-loop-closure \
  --save-world-points \
  --stride 2 --dense-stride 4
```

The ~6.71 GiB working set fits Orin unified memory (it is independent of clip
length — the context window is fixed at 12 frames), but throughput will be far
below the 24.45 FPS H100 benchmark. Thin the stream with `--stride` and keep
dense outputs sparse with `--dense-stride`.

## 3. Export a browser-ready cloud

```bash
python scripts/export_reconstruction_ply.py \
  --poses outputs/g1/camera_poses.npy \
  --points outputs/g1/local_points.pt \
  --colors outputs/g1/colors.pt \
  --metadata outputs/g1/metadata.json \
  --confidence outputs/g1/confidence.pt --confidence-threshold 0.3 \
  --output outputs/g1/reconstruction.ply \
  --bev-output outputs/g1/trajectory_bev.png \
  --point-stride 4 --max-points 2000000
```

`reconstruction.ply` is `binary_little_endian` with `float x/y/z` +
`uchar red/green/blue` vertices. God's Eye View parses that directly in the
browser (`src/data/plyPointCloud.js`) and renders it with a Cesium
`PointPrimitiveCollection`, so the cloud never goes through the telemetry
ingest endpoint or its size cap. Serve `reconstruction.ply` and
`camera_poses.npy` from that app's `public/recon/<clip>/`.

## 4. Cleanup checklist

- [ ] Recording process stopped; `/tmp/g1-clip` (and any staged checkpoint) removed from the Orin.
- [ ] Robot left in its normal mode (developer mode should never have been entered).
- [ ] `conda env remove -n <yourname>` on the Orin.
- [ ] Anything unexpected reported to the admin.
