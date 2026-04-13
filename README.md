# CellTrajectoryAnalyzer

<div align="center">

Cell tracking, trajectory correction, and histogram-based population analysis for microscopy videos of flowing cells.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Tracking%20%26%20Video-0C7C59?style=for-the-badge&logo=opencv&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Numerics-1F4E79?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data%20Tables-1D3557?style=for-the-badge&logo=pandas&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Scientific%20Plots-BC4749?style=for-the-badge">
  <img alt="SciPy" src="https://img.shields.io/badge/SciPy-Fitting%20%26%20Analysis-7B2CBF?style=for-the-badge&logo=scipy&logoColor=white">
</p>

</div>

This repository brings together three connected workflows:

1. Cell detection and tracking in video frames.
2. Trajectory filtering and tilt correction.
3. Histogram-based population analysis and trajectory classification.

The codebase is organized around reusable modules for tracking (`tracking/`), trajectory visualization and correction (`trajectoryplot/`), and downstream deflection analysis (`defanalysis/`).

## Demo Preview

The project now includes an inline GIF preview plus the original recorded demo video:

- ![Cell flow preview](assets/video/cell-flow-preview.gif)
- [Watch the cell-flow video](assets/video/cell-flow-preview.mp4)

The GIF gives a quick visual of cells moving through the channel, while the MP4 keeps the higher-quality full recording available in the repo.

## What This Project Does

- Tracks cells across frames using either a centroid tracker or `SORT`
- Uses `OpenCV` preprocessing steps such as CLAHE, background subtraction, blurring, thresholding, and contour detection
- Saves annotated tracking video plus frame-by-frame CSV outputs
- Filters short trajectories and generates trajectory visualizations
- Supports tilt correction using Sobel, Scharr-RANSAC, ridge-based, zonal, or manual workflows
- Compares experiment and control populations with histogram-based gating
- Produces trajectory classifications, curvature plots, fit diagnostics, and summary reports

## Core Technologies

| Area | Libraries / Tools |
| --- | --- |
| Video I/O and image processing | `OpenCV` |
| Numerical computation | `NumPy` |
| Tables and CSV handling | `Pandas` |
| Plotting and scientific visualization | `Matplotlib` |
| Curve fitting and statistics | `SciPy` |
| Image utilities | `scikit-image` |
| Multi-object tracking | `SORT`, `filterpy` |
| Project code | `tracking/`, `trajectoryplot/`, `defanalysis/` |

## Pipeline Overview

```mermaid
flowchart LR
    A[Microscopy Video] --> B[Frame Preprocessing<br/>CLAHE, subtraction, blur, threshold]
    B --> C[Cell Detection<br/>Contours plus enclosing circles]
    C --> D[Tracking<br/>SORT or centroid tracker]
    D --> E[Trajectory CSVs<br/>IDs, frame counts, radii]
    E --> F[Trajectory Filtering<br/>Minimum frame threshold]
    F --> G[Tilt Correction<br/>Sobel, Scharr, ridge, manual]
    G --> H[Population Analysis<br/>Fits, gating, classification]
    H --> I[Plots, reports, and annotated outputs]
```

## Repository Layout

```text
CellTrajectoryAnalyzer/
|- tracking/          # tracking pipeline, preprocessing, detection, visualization
|- trajectoryplot/    # trajectory filtering, plotting, tilt correction
|- defanalysis/       # histogram gating, fit diagnostics, classification, reporting
|- assets/            # README media and plot showcase placeholders
|- your_video/        # sample output CSVs
`- main.py            # current orchestration script
```

## Quick Start

Install the main dependencies:

```bash
pip install numpy pandas matplotlib scipy scikit-image opencv-python filterpy
```

Then run the pipeline from `main.py` after updating the hard-coded input paths to your own video locations:

```bash
python main.py
```

Typical outputs from the current workflow include:

- Annotated tracking video such as `Recorded_Annotated.avi`
- Sample frames such as `OriginalFrame.png`, `GrayImage.png`, and `Threshold.png`
- Trajectory plots such as `Original_trajectories_plot.png` and `filtered_trajectories_corrected_plot.png`
- Population analysis plots such as `Histogram_Analysis_StaticGate.png` and `Trajectories_Classified.png`

## Example Visual Outputs

These are the actual trajectory and histogram figures currently stored in `assets/`.

| Trajectory overview | Tilt-corrected trajectories |
| --- | --- |
| ![Original trajectories](assets/plots/trajectories/Original_trajectories_plot.png) | ![Corrected trajectories](assets/plots/trajectories/filtered_trajectories_corrected_plot.png) |

| Classified trajectories | Histogram-based gating |
| --- | --- |
| ![Classified trajectories](assets/plots/trajectories/deflection_classified_trajectories.png) | ![Static gate histogram](assets/plots/histograms/Histogram_Analysis_StaticGate.png) |

## Notes

- `tracking/pipeline.py` is the main entry point for video tracking and annotated output generation.
- `trajectoryplot/trajectory.py` handles trajectory filtering, tilt estimation, and corrected trajectory plots.
- `defanalysis/deflection_analysis.py` handles gating, fit diagnostics, histogram analysis, and classification outputs.
- The `assets/` folder now contains the tracked-cell video preview plus representative trajectory and histogram outputs used by this README.
