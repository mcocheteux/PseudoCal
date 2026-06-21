"""
PseudoCal: initialisation-free camera-LiDAR extrinsic calibration.

A reference implementation of PseudoCal (Cocheteux, Moreau & Davoine, BMVC 2023;
arXiv:2309.09855) by one of the paper's authors, built on top of the UniCal++ package
(https://github.com/mcocheteux/unical-plus). PseudoCal is a three-stage cascade:

    1. PseudoPillars — a coarse, initialisation-free stage that lifts the camera
       image to a 3-D *pseudo-LiDAR* cloud (via monocular metric depth), encodes
       it together with the real LiDAR using a PointPillars encoder, and fuses the
       two bird's-eye-view pseudo-images with a MobileViT backbone.
    2. UniCal-M — medium-range residual refinement (the UniCal++ model).
    3. UniCal-S — fine-range residual refinement (the UniCal++ model).

Stages 2 and 3 are reused directly from ``unical``; only the PseudoPillars stage
and the cascade orchestration are implemented here.
"""

__version__ = "0.1.0"
