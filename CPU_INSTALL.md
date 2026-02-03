CPU-only PyTorch install instructions

Windows (recommended):
```
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio --extra-index-url https://pypi.org/simple
```

Linux (recommended):
```
python3 -m pip install --upgrade pip
pip3 install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio --extra-index-url https://pypi.org/simple
```

If you prefer a specific version, add `==<version>` after `torch` (for example `torch==2.1.2+cpu`). Using the PyTorch CPU wheels ensures no CUDA drivers or GPU libs are required.

spaCy (optional, improves negation scope detection):
```
pip install spacy
python -m spacy download en_core_web_sm
```
