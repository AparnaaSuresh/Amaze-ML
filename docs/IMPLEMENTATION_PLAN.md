# Implementation Plan

The pipeline implements the maximum-efficiency plan in seven gated stages:
EDA and exact metric; deterministic baseline; candidate frontier; hard-negative
LightGBM matcher; F0.5 policy; justified advanced retrieval; final package.

The champion is the smallest candidate policy within one standard error of the
best three-fold S1-level macro-F0.5 mean. Every selected run must preserve its
resolved configuration, fold assignment, candidate report and validator output.

Use `pipeline.py` commands from the repository root. Do not use external
business data, lookup, geocoding, APIs, or services.
