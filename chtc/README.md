# Study 2 on CHTC (HTCondor)

Short README on how to run **Study 2** (BioViL-T inference over MIMIC-CXR temporal pairs) on CHTC.

## Prerequisites

- A **CHTC account** and access to the **GPU Lab** and **staging** (`HasCHTCStaging`). Confirm policies with [CHTC GPU jobs](https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs) and [Docker jobs](https://chtc.cs.wisc.edu/uw-research-computing/docker-jobs).
- **Docker Hub** (or another registry HTCondor can pull from): you will build and push an image built from `Dockerfile`.
- **MIMIC data** prepared locally under `data/` with the paths expected by `stage_data.sh` (appropriate credentialing and DUA compliance).
- **First-run model weights**: BioViL-T is loaded via `hi-ml-multimodal` and may download from Hugging Face on the worker. Workers typically need outbound network for that first fetch, unless you bake a cache into the image.

## 1. Build and push the container image

From the **repository root** (parent of `chtc/`):

```bash
docker build -f chtc/Dockerfile -t YOURDOCKERHUB/cs767_study2:latest .
docker push YOURDOCKERHUB/cs767_study2:latest
```

Edit `study2.sub` and set `container_image` to your image URI.

## 2. Stage the dataset tarball

On a machine that has the Study 2 inputs (metadata CSV + MIMIC-CXR-JPG tree), from the repo root:

```bash
bash chtc/stage_data.sh
```

By default this writes `chtc/study2_data.tar.gz`. The archive contains:

- `data/MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv`
- `data/MIMIC-CXR-JPG/files/` (full tree — **very large**; stage only what you need for tests)

Upload the tarball to **CHTC staging** (for example via `transfer.chtc.wisc.edu`) under your netid, e.g. `/staging/<netid>/study2_data.tar.gz`.

In `study2.sub`, set:

- `netid` to your NetID  
- `input_staging_path` to `file:///staging/<netid>/<your-file>.tar.gz`  
- `input_tarball_name` to the **basename** of that file (what the worker sees after transfer), e.g. `study2_data.tar.gz`

Adjust `transfer_output_remaps` if you want results written to a different staging path than the default in the file.

## 3. Submit the job

Use a clone of this repo on the CHTC submit node (e.g. `submit1.chtc.wisc.edu`), with `chtc/` at the project root as in the repo layout.

```bash
mkdir -p chtc/logs
condor_submit chtc/study2.sub
```

### Common overrides

```bash
condor_submit chtc/study2.sub \
  container_image=docker://YOURDOCKERHUB/cs767_study2:v2 \
  input_staging_path=file:///staging/YOURNETID/study2_data.tar.gz \
  input_tarball_name=study2_data.tar.gz \
  run_suffix=mypilot \
  netid=YOURNETID
```

Reduce GPU requirements if your slice fits smaller cards (see `gpus_minimum_memory` and resource requests in `study2.sub`).

## 4. What runs inside the job

- **`run_study2.sh`** is the HTCondor **executable**. It unpacks the staged tarball into scratch, runs:

  `python code/study2/scripts/run_inference.py`  

  with `--device cuda` by default, then packs outputs into **`study2_bundle.tar`** for transfer back.

- Logs go under **`chtc/logs/`** on the submit side (`study2_<Cluster>.log`, `.out`, `.err`).

## 5. Environment variables (optional)

With `getenv = True`, variables you export on the submit host can affect the job (confirm what your submit environment forwards).

| Variable | Purpose |
|----------|---------|
| `STUDY2_DEVICE` | Force device (default behavior uses CUDA in the script). |
| `STUDY2_MAX_PAIRS` | If set, passes `--max-pairs` for a **pilot** subset. |

Example pilot (combine with submit-node export or your preferred HTCondor `environment` line):

```bash
export STUDY2_MAX_PAIRS=500
condor_submit chtc/study2.sub
```

## 6. Outputs

- **Transferred back**: `study2_bundle.tar` (contents: `study2_out/<run_suffix>/` with CSV, embeddings `.npz`, and QC JSON — same artifacts as local `run_inference.py`).
- **Remapped to staging** (per `transfer_output_remaps` in `study2.sub`): typically  
  `/staging/<netid>/cs767_project/<run_suffix>.tar` — adjust the remap path in `study2.sub` to match where you want artifacts on staging.

## Files in this folder

| File | Role |
|------|------|
| `study2.sub` | HTCondor submit description |
| `run_study2.sh` | Entry script inside the container |
| `stage_data.sh` | Build `study2_data.tar.gz` from local `data/` |
| `Dockerfile` | Image definition for Study 2 |
| `logs/` | Directory for HTCondor `.log`, `.out`, `.err` paths |
