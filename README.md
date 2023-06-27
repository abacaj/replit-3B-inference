# Replit Code Instruct inference code using CPU

Run inference on the replit code instruct using your CPU. This inference code uses a [ggml](https://github.com/ggerganov/llama.cpp) quantized model. To run the model we'll use a library called [ctransformers](https://github.com/marella/ctransformers) that has bindings to ggml in python.

Demo:

[Inference Demo](null)

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

Next install dependencies.

```sh
pip install -r requirements.txt
```

Next download the quantized model weights (about 19GB).

```sh
python download_model.py
```

Ready to rock, run inference.

```sh
python inference.py
```

Next modify inference script prompt and generation parameters.
