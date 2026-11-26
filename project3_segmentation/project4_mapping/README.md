# Project 4: End-to-End Drone Mapping & AI Pipeline

**Course:** AI-driven Perception, Learning and Mapping for Drones — Modules 5–6
**Status:** Complete
**Date:** November 2026

## Summary

This project is the capstone pipeline: taking raw drone images, generating a 3D reconstruction and orthomosaic (via photogrammetry), running a custom AI segmentation model on the resulting map, and exporting a client-ready GIS deliverable.

## The Pipeline

Raw Drone Photos (172 images)
↓
OpenDroneMap (WebODM) — Photogrammetry with GCPs
↓
Orthomosaic (11,390 x 9,255 px GeoTIFF) + Digital Surface Model (DSM)
↓
PyTorch U-Net — Tiled inference (512x512 patches via Rasterio)
↓
Georeferenced Mask (mask.tif)
↓
QGIS — Raster to Vector Polygonize + Cartographic Layout
↓
Client Deliverable (PDF Map + GeoPackage)


## Results

- **Photogrammetry:** 172 images processed in 9.7 hours. Average Ground Sample Distance (GSD) of **2.62 cm/px** (survey-grade resolution). Georeferencing achieved via Ground Control Points (GCP).
- **AI Segmentation:** Successfully tiled, inferred, and stitched a 209 MB orthophoto without running out of RAM. (Note: the AI model used in this test was trained for 1 epoch to demonstrate the pipeline; the grid artifacts in the mask reflect the 512x512 tile boundaries).
- **GIS Output:** Converted AI raster predictions into vector polygons using QGIS. Exported a polished PDF layout and a georeferenced map.

## Repository Contents

- `mask_pipeline.py` — Python script for reading GeoTIFFs, tiling, running PyTorch inference, and stitching masks.
- `sheffield_map.png` — The final cartographic layout generated in QGIS.
- `sheffield_map.pdf` — The client-ready PDF deliverable.

## Key Takeaways

1. **Photogrammetry works best with GCPs.** Relying purely on GPS EXIF data leads to map drift. Using Ground Control Points ensured centimeter-level accuracy.
2. **Chunking is essential for large maps.** A 209 MB GeoTIFF cannot be pushed through a neural network in one pass. Using `rasterio.windows` to process 512x512 tiles preserves memory.
3. **The gap between AI and GIS is vectors.** Clients don't want pixels; they want polygons. QGIS's `Polygonize` tool bridges this gap seamlessly.

## How to Reproduce

1. **WebODM:** Process images with the "High Resolution" preset and upload the `gcp_list.txt`.
2. **Colab:** Run `mask_pipeline.py` to generate the mask.
3. **QGIS:** Load the orthophoto and mask, run `Raster -> Conversion -> Polygonize`, and build the print layout.

