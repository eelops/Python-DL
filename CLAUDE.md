# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep learning course homework repository (Chinese-language), following a curriculum similar to "Dive into Deep Learning" (d2l.ai). Contains homework templates, completed solutions, and study notes.

## Repository Structure

- `original/` — Blank homework templates as distributed (TODOs unfilled)
- `tutorial/` — Completed solutions with all cells executed
- `notes/` — Encyclopedia-style reference notes (Chinese), one per homework (e.g., `HW1_notes.md` ↔ `HW1.ipynb`)

## Environment

- Anaconda environment: `deeplearning`
- Activate: `conda activate deeplearning`
- Run notebook: `jupyter notebook` (select `deeplearning` kernel)
- Dependencies (implicit, no requirements.txt): `torch`, `pandas`, `numpy`
- Notebooks must be run with cells executed **in order** from top (import cells first, otherwise `NameError`)

## Conventions

- All prose, markdown cells, and notes are written in **Chinese**
- Homework exercises use `# TODO:` comments marking where to fill in code
- `original/` preserves blank TODOs; `tutorial/` has completed answers
- HW exercises progress in difficulty: HW1 = pandas + tensor basics, HW2 = linear algebra operations

## Working with Notebooks

- When filling in TODOs: only add the required code, do not modify surrounding scaffold or print statements
- When generating notes: write comprehensive, encyclopedia-style references covering every keyword/API used in the corresponding homework, in Chinese, pure substance with no filler
