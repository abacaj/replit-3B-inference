# Replit Code Instruct inference using CPU

Run inference on the replit code instruct model using your CPU. This inference code uses a [ggml](https://github.com/ggerganov/llama.cpp) quantized model. To run the model we'll use a library called [ctransformers](https://github.com/marella/ctransformers) that has bindings to ggml in python.

Demo:

[Inference Demo](https://github.com/abacaj/replit-3B-inference/assets/7272343/a68ec17a-830b-4d76-9df2-166ca6b7fb2b)

## Requirements

Using docker should make all of this easier for you. Minimum specs, system with 8GB of ram. Recommend to use `python 3.10`.

## Tested working on

Will post some numbers for these two later.

- AMD Epyc 7003 series CPU
- AMD Ryzen 5950x CPU

## Setup

First create a venv.

```sh
python -m venv env && source env/bin/activate
```
Next install the submodule with ctransformers change.

```sh
git submodule update --init --recursive
```

Next install dependencies.

```sh
pip install -r requirements.txt
```

Next download the quantized model weights (about 1.5GB).

```sh
python download_model.py
```

Ready to rock, run inference.

```sh
python inference.py
```

Next modify inference script prompt and generation parameters.
