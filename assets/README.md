# Assets Guide

This folder is meant for presentation-ready media used in the top-level `README.md`.

## Suggested Structure

```text
assets/
|- video/
|  `- cell-flow-preview.svg
`- plots/
   |- trajectories/
   |  |- original-trajectories.svg
   |  |- corrected-trajectories.svg
   |  `- classified-trajectories.svg
   `- histograms/
      |- static-gate-histogram.svg
      `- feature-gate-histogram.svg
```

## Recommended Replacements

When you are ready to showcase real outputs, either replace the placeholder assets with your own media or update the paths in `README.md`.

- Tracking video preview:
  Save an animated GIF to `assets/video/cell-flow-preview.gif`
- Full tracking video:
  Save a playable video to `assets/video/cell-flow-demo.mp4`
- Original trajectories:
  Copy `Original_trajectories_plot.png` to `assets/plots/trajectories/`
- Corrected trajectories:
  Copy `filtered_trajectories_corrected_plot.png` to `assets/plots/trajectories/`
- Classified trajectories:
  Copy `Trajectories_Classified.png` to `assets/plots/trajectories/`
- Main histogram:
  Copy `Histogram_Analysis_StaticGate.png` to `assets/plots/histograms/`
- Extra feature histogram:
  Copy one of the fit-specific or curvature histograms from your output folders into `assets/plots/histograms/`

## Good Candidates From Your Pipeline

Useful outputs already produced by this repository include:

- `Recorded_Annotated.avi`
- `OriginalFrame.png`
- `GrayImage.png`
- `Threshold.png`
- `Original_trajectories_plot.png`
- `filtered_trajectories_plot.png`
- `filtered_trajectories_corrected_plot.png`
- `Trajectories_Classified.png`
- `Histogram_Analysis_StaticGate.png`

Keeping README assets in this folder makes it easier to separate polished showcase media from raw experimental output directories.
