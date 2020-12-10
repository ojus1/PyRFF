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

# Array of Fixed size vectors
non_sequential = np.random.normal(size=(4, 5))
feat = get_features(
    non_sequential,
    123,
    "rff",
    10,
    0.1
)

print(feat.shape)  # (4, 20)
