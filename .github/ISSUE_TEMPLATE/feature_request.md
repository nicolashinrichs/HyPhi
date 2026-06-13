---
name: Feature request
about: Propose a new capability or an extension to HyPhi
title: ""
labels: enhancement
assignees: ""
---

## Summary

A clear, one or two sentence description of the proposed capability.

## Motivation

What scientific or practical need this serves, and why it belongs in HyPhi. If it adds a new
method (a curvature notion, an entropy estimator, a connectivity measure, a modality), name the
defining reference so the implementation can be grounded in it.

## Roadmap pillar

Which area of the roadmap this fits (delete the ones that do not apply):

- Curvature and flow core
- Entropy suite and numerical safety
- Public API, docs, and reproducibility
- Modalities and connectivity adapters (EEG, MEG, fNIRS, fMRI, HyPyP interop)
- Statistics, nulls, and group generalization
- Benchmarks, ground truth, and the comparison matrix
- Visualization and embeddings
- Engineering infrastructure (CI, packaging, pipeline)

## Proposed scope

What the change adds, where it lives (which module under `code/hyphi/`), and how it connects to
the existing pipeline (for a modality, the connectivity matrix it produces feeds
`analyses.build_sliding_window_graphs`). Keep HyPhi complementary: prefer orchestrating HyPyP,
MNE, and NetworkX over reimplementing them.

## Acceptance criteria

How a reviewer will know it is done: the validation, the test, the tutorial, the public API
addition.
