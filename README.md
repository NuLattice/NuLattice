# `NuLattice`: Ab initio computations of atomic nuclei on lattices

`NuLattice` is a set of Python programs (full configuration interaction,
coupled cluster method, Hartree Fock, in-medium similarity
renormalization group) for computations of atomic nuclei on spatial
lattices with periodic boundary conditions. At present, the available
Hamiltonians are based on pionless effective field theory and consist
of the kinetic energy, an on-site (SU4 symmetric) two-body contact,
and an on-site three-body contact.

[![Documentation Status](https://app.readthedocs.org/projects/nulattice/badge/)](https://nulattice.readthedocs.io/)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17094172.svg)](https://doi.org/10.5281/zenodo.17094172)
[![arXiv](https://img.shields.io/badge/arXiv-2509.08771-b31b1b.svg)](https://arxiv.org/abs/2509.08771)

## Setup

To set up the repository to run the code, install the required packages in `requirements.txt`.

We recommend setting up a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
This virtual environment can always be (re)activated with:
```
source .venv/bin/activate
```

Once the virtual environment has been set up, 
the scripts in `examples` and `benchmarks` can simply be run as (for example):
```
python3 Example_CCSD.py
```

## Contributing

To contribute to `NuLattice`. please fork the repository, 
create a new branch with your feature, 
and once it is complete open a pull request to contribute to the main project.
See [Contributing](CONTRIBUTING.md) for more details, including how to set up a private development fork of `NuLattice`.
Consistent contributors will be given the option to also serve as maintainers of the project.

## Citing

To cite `NuLattice` in your research, please reference:

M. Rothman, B. Johnson-Toth, G. Hagen, M. Heinz, T. Papenbrock, `NuLattice`: Ab initio computations of atomic nuclei on lattices, [arXiv:2509.08771](https://arxiv.org/abs/2509.08771)

For BibTex users, we provide the BibTex entry:

```
@misc{Rothman2025NuLattice,
      title={{N}u{L}attice: Ab initio computations of atomic nuclei on lattices}, 
      author={M. Rothman and B. Johnson-Toth and G. Hagen and M. Heinz and T. Papenbrock},
      year={2025},
      eprint={2509.08771},
      archivePrefix={arXiv},
      primaryClass={nucl-th},
      url={https://arxiv.org/abs/2509.08771},
      doi={10.48550/arXiv.2509.08771}
}
```

