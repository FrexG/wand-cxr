
# WAND Framework

WAND (Wasserstein-based ANatomical Distance) is a general framework for morphometric evaluation consisting of:

1. Segmentation
2. Feature Extraction
3. Distribution Construction
4. Wasserstein Comparison

This repository provides an instantiation of WAND for chest radiography (WAND-CXR).

## Usage

```bash
python main.py \
  --csv_r real_images.csv \
  --csv_s synthetic_images.csv \
  --num 500 \
  --output_dir output
```