## PyRFF - Random Fourier Features with Numpy

### About - TLDR

- Use this repo if you're too lazy to extract features
- But still want good performance (in terms of speed and model "accuracy")
- The only dependency is NumPy.
- Extremely Lightweight
- Use this for Millions of predictions/day.

This repository extends [this repo](https://github.com/tiskw/Random-Fourier-Features) to sequential features.

## Usage

```
from PyRFF import get_features_sequential, get_features
import numpy as np

# List of variable size vectors
sequential = [np.random.normal(size=(np.random.randint(1, 12), 4))
              for i in range(4)]
# Get Sequential Features
feat = get_features_sequential(
    sequential,  # Input List
    123,  # Random Seed
    # Feature Type orf (Orthogonal Random Feature) or rff (Random Fourier Features)
    "orf",
    6,  # Output Feature Size // 2
    0.1,  # Standard Deviation for Random Kernel
    max_length=6  # Maximum padded size for input vector (time dimension)
)

print(feat.shape)  # (4, 12)
```

## Installation

`pip install PyRFF`

- Check (Example.py)[./Example.py] for the starter code.

## Authors

### This Repo

Surya Kant Sahu (surya.@gmail.com)[surya.@gmail.com]

### Original Author

Tetsuya Ishikawa (tiskw111@gmail.com)[tiskw111@gmail.com]
