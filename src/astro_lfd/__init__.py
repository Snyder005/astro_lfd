"""astro_lfd: Linear Feature Detection for LSST Camera images.

Detects linear features (satellite streaks, cosmic-ray tracks, diffraction
spikes) in astronomical images. Detectors are formulated as standard LSST tasks
sharing a common template, differing only in the line-finding core: the Kernel
Hough Transform (KHT) detector, an Approximate Discrete Radon Transform (ADRT)
detector, and further methods (e.g. Line Segment Detector, Frangi vesselness,
YOLO) that can be added behind the same interface.
"""
