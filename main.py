import cv2
import numpy as np
import glob
import yaml

CHECKERBOARD = (7, 5)
SQUARE_SIZE = 30  # mm
IMAGE_PATH = "images/*.jpg"
OUTPUT_FILE = "config.yaml"

subpix_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.1
)

calibration_flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
    cv2.fisheye.CALIB_FIX_SKEW
)

# PREPARE OBJECT POINTS

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = (
    np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
    .T.reshape(-1, 2) * SQUARE_SIZE
)

objpoints = []
imgpoints = []

# LOAD IMAGES

images = glob.glob(IMAGE_PATH)

if len(images) == 0:
    raise RuntimeError("No images found")

print("Found images:", len(images))

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD)

    if ret:

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (3, 3),
            (-1, -1),
            subpix_criteria
        )

        imgpoints.append(corners2)

        print("Detected:", fname)

    else:
        print("FAILED:", fname)

# CALIBRATION

N_OK = len(objpoints)

print("Valid images:", N_OK)

DIM = gray.shape[::-1]

K = np.zeros((3, 3))
D = np.zeros((4, 1))

rvecs = []
tvecs = []

rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    DIM,
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print("RMS:", rms)
print("K:\n", K)
print("D:\n", D)

# SAVE YAML

data = {
    "image_width": int(DIM[0]),
    "image_height": int(DIM[1]),
    "K": {
        "rows": 3,
        "cols": 3,
        "dt": "d",
        "data": K.reshape(-1).tolist()
    },
    "D": {
        "rows": 1,
        "cols": 4,
        "dt": "d",
        "data": D.reshape(-1).tolist()
    }
}

with open(OUTPUT_FILE, "w") as f:
    yaml.dump(data, f)

print("Saved:", OUTPUT_FILE)
