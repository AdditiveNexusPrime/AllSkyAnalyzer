# AllSkyAnalyzer

**AllSkyAnalyzer** is a Python application that analyses images captured by an
allsky camera (indi-allsky) to:

* **Detect and label known stars** from a built-in photometric catalogue.
* **Draw constellation lines** when a plate-solve solution is available.
* **Flag anomalies** (meteors, satellites, artefacts) using GPU-accelerated
  statistical background subtraction.
* **Save annotated overlay images** and JSON sidecar files alongside each
  capture.

It is designed to run as a Docker container on an Ubuntu server with NVIDIA
GPUs and integrates directly with a local
[indi-allsky](https://github.com/aaronwmorris/indi-allsky) installation.

---

## Architecture

```
AllSkyAnalyzer/
├── main.py                  # CLI entry-point (analyze / watch / serve)
├── src/
│   ├── config.py            # YAML-backed configuration dataclasses
│   ├── indi_allsky_reader.py# Discovers images from indi-allsky directories
│   ├── star_finder.py       # Photometric detection + plate-solve + catalogue
│   ├── anomaly_detector.py  # GPU-accelerated z-score anomaly detection
│   ├── overlay.py           # Draws annotations onto images
│   ├── analyzer.py          # Orchestrates the full pipeline
│   ├── watcher.py           # Filesystem watcher (triggers on new images)
│   └── api.py               # FastAPI REST interface
├── config/
│   └── config.yaml          # Default configuration (copy and customise)
├── tests/                   # pytest unit tests
├── Dockerfile               # NVIDIA CUDA 12.4 + Python 3.11
└── docker-compose.yml       # Watcher + API server services
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Docker | ≥ 24 |
| Docker Compose | v2 |
| NVIDIA Container Toolkit | latest |
| NVIDIA GPU | Any CUDA 12-capable card (tested on RTX 2060) |
| indi-allsky | Any recent version |

---

## Quick Start (Docker)

### 1. Clone & configure

```bash
git clone https://github.com/AdditiveNexusPrime/AllSkyAnalyzer.git
cd AllSkyAnalyzer
cp config/config.yaml config/local.yaml
# Edit config/local.yaml to match your indi-allsky paths
```

### 2. Build the image

```bash
docker compose build
```

### 3. Run the watcher + REST API

```bash
docker compose up -d
docker compose logs -f
```

The watcher will start monitoring the indi-allsky image directory and
analysing every new frame.  The REST API is available at
`http://localhost:8000`.

### 4. One-shot analysis

```bash
docker run --gpus all \
    -v /var/lib/indi-allsky:/var/lib/indi-allsky:ro \
    -v /data/output:/data/output \
    allskyanalyzer:latest \
    analyze /var/lib/indi-allsky/images/latest.jpg
```

---

## Local Development

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the CLI

```bash
# Analyse a single image
python main.py analyze /path/to/image.jpg --output-dir ./output

# Watch a directory
python main.py watch

# Start the REST API server
python main.py serve
```

### Run tests

```bash
pytest
```

---

## Configuration Reference

The application is configured via a YAML file (default:
`config/config.yaml`).  Pass an alternative path with `--config`.

| Section | Key | Default | Description |
|---|---|---|---|
| `indi_allsky` | `image_dir` | `/var/lib/indi-allsky/images` | Directory scanned for images |
| `indi_allsky` | `file_pattern` | `**/*.jpg` | Glob pattern for image files |
| `indi_allsky` | `max_age_seconds` | `0` | Ignore images older than N seconds (0 = all) |
| `astrometry` | `index_dir` | `/usr/share/astrometry` | Local astrometry.net index path |
| `astrometry` | `magnitude_limit` | `6.0` | Faintest catalogue stars to overlay |
| `anomaly` | `z_score_threshold` | `5.0` | Sensitivity (lower = more detections) |
| `anomaly` | `use_cuda` | `true` | Enable GPU acceleration |
| `overlay` | `star_colour` | `[255,255,0,200]` | RGBA colour for star circles |
| `overlay` | `anomaly_colour` | `[255,50,50,220]` | RGBA colour for anomaly boxes |
| `output` | `directory` | `output` | Where to write overlays and JSON |

---

## REST API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/config` | GET | Current configuration |
| `/analyze` | POST | Upload an image, receive JSON results |
| `/latest` | GET | Analyse the most-recent indi-allsky image |
| `/overlay/{filename}` | GET | Download a saved overlay image |

---

## GPU Support

The application uses two NVIDIA RTX 2060 GPUs.  Both are exposed to the
containers via the `count: all` setting in `docker-compose.yml`.  PyTorch
operations and OpenCV CUDA kernels will automatically use the GPU specified
by `anomaly.cuda_device` (default: `0`).

Set `CUDA_VISIBLE_DEVICES=0,1` in the container environment to expose both
cards, then set `cuda_device: 1` in the config to use the second GPU.

---

## Anomaly Detection

The detector uses **local background subtraction**:

1. Convert the image to grayscale.
2. Estimate the background with a large Gaussian blur.
3. Subtract the background; compute the standard deviation of the residual.
4. Flag pixels more than `z_score_threshold` σ above the mean.
5. Extract contours; filter by area to remove noise and large clouds.
6. Classify each region as `point`, `streak` (meteor / satellite), or
   `diffuse`.

---

## Star & Constellation Identification

Without a plate-solve solution:
* OpenCV `SimpleBlobDetector` finds bright point sources.

With `astrometry.net` installed (`astrometry-data-tycho2` package):
* `solve-field` determines the sky coordinates (RA/Dec) of the image centre.
* Catalogue stars are projected into pixel space and cross-matched with
  detected blobs.
* Constellation line segments are drawn using gnomonic (TAN) projection.

---

## License

MIT
