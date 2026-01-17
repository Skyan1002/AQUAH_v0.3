# AQUAH v0.3

AQUAH is a Level-4 Hydrologic AI System for end-to-end flash-flood analysis. It automates basin selection, precipitation/PET ingestion, CREST/EF5 execution, visualization, and optional report generation for hydrologic investigations.

## What It Does

- Parses natural-language event descriptions to identify basins and time windows.
- Downloads and processes MRMS precipitation and PET forcing data.
- Runs EF5/CREST simulations and compares simulated discharge to USGS observations.
- Produces maps and hydrographs for basin characterization and model evaluation.
- (Optional) Generates a publication-ready PDF report with diagnostic interpretation.

## Quick Start (Local)

1. Install dependencies listed in `requirements.txt`.
2. Run with the provided sample arguments:

```bash
python main.py @args.txt
```

## Report Generation (PDF)

Report generation is **disabled by default**. Enable it with `--report` to produce a PDF report after the simulation completes:

```bash
python main.py @args.txt --report
```

The PDF will be saved under `report/<basin>_<timestamp>/` with a name like:

```
Hydro_Report_<Basin>_<Model>_00.pdf
```

## Docker

You can also run AQUAH via Docker using the image:

```
skyan1002/aquah
```

Example:

```bash
docker pull skyan1002/aquah

docker run --rm -it \
  -e OPENAI_API_KEY=your_key_here \
  -v $(pwd):/workspace/AQUAH_v0.3 \
  skyan1002/aquah \
  python main.py @args.txt --report
```

## Configuration Notes

- `args.txt` contains a sample configuration for running a flash-flood analysis.
- Figures are saved under `figures/<timestamp>/`.
- Reports are saved under `report/<basin>_<timestamp>/` when `--report` is enabled.
