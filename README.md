# HTucker: A python repository for hierarchical Tucker decomposition

This repository contains python implementations for computing an approximation of multidimensional arrays in hierarchical Tucker format ([Grasedyck 2010](https://doi.org/10.1137/090764189), [Oseledets 2009](https://doi.org/10.1137/090748330), [Kressner 2014](https://sma.epfl.ch/~anchpcommon/publications/htucker_manual.pdf)). In addition to that, this repository contains python implementations of 2 new algorithms, `BHT-l2r` (Batch Hierarchical Tucker - leaves to root) and `HT-RISE` (Hierarchical Tucker - Rapid Incremental Subspace Expansion).

`BHT-l2r` computes an approximation of a batch of tensors in batch hierarchical Tucker format, a batch-modified version of the hierarchical Tucker format, which provides a tensor network structure more suitable for incremental updates. `HT-RISE` incrementally updates an existing approximation in batch hierarchical Tucker format and (to the best of our knowledge) is the first algorithm that updates an existing approximation in hierarchical Tucker format. Both algorithms provide mathematically proven approximation error upper bounds.

## Installation
I suggest you create a python virtual environment. Then within that environment you can install an editable installation with
```
python -m pip install -e .
```
a regular install would remove the `-e` term.

## Running

The unit tests are still under development. However, the current version of the tests should still pass without any issues.
You should be able to execute
```
python -m unittest tests/main_test.py -v
```
and see all tests pass. Please report through an issue if you have any problems.

## Development

Feel free to request/suggest new functionalities to the package. There is a plan to combine this repository with the [`TT-ICE`](https://github.com/dorukaks/TT-ICE) repository to create a holistic incremental tensor decomposition package. However, the development plan is still TBD.

If you want to contribute to the package, feel free to create your branch and open a PR. I will do my best to review any PRs in a timely manner.

### Development Setup

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on setting up your development environment and contributing to this project.

### Linting and Code Quality

To run linting manually:
```
make lint
```

This repository uses pre-commit hooks to ensure code quality. To set up:
```bash
# Install pre-commit
pip install pre-commit

# Set up pre-commit hooks
pre-commit install
```

#### Helpful Scripts

We provide several scripts to help with development:

- `scripts/run-pre-commit-staged.sh` - Run pre-commit only on staged files
- `scripts/demo-pre-commit.sh` - Demonstrate the differences between running pre-commit on all files vs. staged files
- `scripts/troubleshoot-pre-commit.sh` - Diagnose and fix common pre-commit issues

For detailed information about pre-commit, see [docs/pre-commit-guide.md](docs/pre-commit-guide.md).

## POTENTIAL ISSUES:
0. currently edges are not actually used to determine contraction path, just index names. This causes issues when contracting when attaching tensors of edges are the same
1. benchmarking done but dimension scaling is off (possibly because path is not optimal). Could hardcode optimal TT path, but it should be ok for lower dimensions (lower number of cores) May be a problem if QTT considered in the future

## Funding
AFOSR Computational Mathematics Program under the Award \#FA9550-24-1-0246
Schmidt Sciences, LLC.,

## Authors
Alex Gorodetsky, goroda@umich.edu
Zheng Guo, zhgguo@umich.edu
Aditya Deshpande, dadity@umich.edu

Copyright 2024-2025, Regents of the University of Michigan

## Features
- **Tensor Decomposition Algorithms:** HT, BHT, and TT decompositions implemented in a modular library (`htucker/`).
- **Incremental & Batch Processing:** Scripts for both incremental and batch tensor compression, suitable for large datasets.
- **Experimentation & Benchmarking:** Ready-to-use scripts for running experiments and generating results for scientific publications.
- **Visualization & Analysis:** Tools for spectrum analysis, error analysis, and visualization of results.
- **Reproducibility:** Integration with [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking and reproducibility.

## Getting Started
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run an example experiment:**
    To run the experiments showcased in the paper `Incremental Hierarchical Tucker Decomposition`, please check the [`README`](./paper_experiments/README.md) file located under the [`./paper_experiments`](./paper_experiments/) folder.
   ```bash
   python paper_experiments/compress_CatGel_TT.py --help
   ```
3. **Explore the library:**
   See [`examples/`](./examples/) for example usages in your own projects.

## Usage Example
```python
from htucker.decomposition import ht_decompose
import numpy as np

tensor = np.random.rand(8, 8, 8, 8)
ht = ht_decompose(tensor, epsilon=1e-4)
reconstructed = ht.reconstruct()
error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
print(f'Relative error: {error}')
```

## Documentation
- Main documentation: [`README.org`](../README.org)
- Scientific paper: [`doc/htucker_paper.pdf`](../doc/htucker_paper.pdf)

## Citing
If you use this codebase in your research, please cite the accompanying paper using the following bibtex entry in your papers:
```bibtex
@article{aksoy2024incremental,
  title={Incremental Hierarchical Tucker Decomposition},
  author={Aksoy, Doruk and Gorodetsky, Alex A},
  journal={arXiv preprint arXiv:2412.16544},
  year={2024}
}
```

## License
This project is licensed under the MIT License. See the [`LICENSE`](./LICENSE) file for details.
