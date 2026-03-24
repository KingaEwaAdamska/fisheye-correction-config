# Fisheye Camera Calibration

Minimal script for fisheye camera calibration using OpenCV.

## Setup
Install dependencies:

`pip install -r requirements.txt `

## Data Preparation
Capture multiple images of a checkerboard pattern.
Place them in the images/ directory.
Adjust parameters in the script if needed:
CHECKERBOARD – number of inner corners (e.g. (7, 5))
SQUARE_SIZE – square size in mm

## Run
`python calibrate.py`

It would generate `config.yaml`
