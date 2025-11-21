# Example calculations using `NuLattice`

This directory contains scripts and IPython notebooks that perform example calculations using `NuLattice` modules.

Scripts can be executed from the project root directory or the `examples` directory by executing, for example,

```
python3 Example_CCSD.py
```

To start the IPython notebook server, run
```
python3 -m jupyter notebook
```
which will open up a window in your browser.
Then navigate to examples and run the notebook you are interested in.

The different examples are briefly explained below.
Systems and lattice sizes have been selected to cover a broad range of cases,
but the interested physicist can easily test other combinations
by changing a few constants at the top of the script
and/or by selecting a different reference state.

## `Example_CCSD` and `Example_CCSD2`

`Example_CCSD` and `Example_CCSD2` compute the ground state of oxygen-16 in the CCSD approximation.
They differ in how the Hamiltonian is constructed, 
with `Example_CCSD` focusing on the pionless effective field theory case
explored in arXiv:2509.08771.
`Example_CCSD2` uses functions
that are extensible to other interactions,
exhibiting how one would perform CCSD computations with improved Hamiltonians.

## `Example_FCI`

`Example_FCI` computes the ground states of the deuteron, helium-3, and helium-4
using exact diagonalization.

## `Example_Hartree_Fock`

`Example_Hartree_Fock` performs a Hartree-Fock computation of oxygen-16.

## `Example_IMSRG`

`Example_IMSRG` performs an IMSRG(2) computation of helium-3.

## Reproduction scripts for publications

### arXiv:2509.08771

Reproduction scripts for selected figures and tables from arXiv:2509.08771 are provided in `reproduction_arxiv_2509.08771`.
