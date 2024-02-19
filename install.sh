#!/bin/bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install ray[tune] ax-platform pytorch-lightning torchmetrics jupyter
