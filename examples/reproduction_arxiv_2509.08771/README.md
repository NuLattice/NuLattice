# Reproduction arXiv:2509.08771

This directory contains two scripts to compute some of the values presented in Fig. 5 and Table 5 of arXiv:2509.08771.
`table_5` computes results presented in Table 5.
IMSRG(2) results are skipped by default because they require 16 GB and 10s of hours to compute.
These can be enabled by changing the `with_imsrg2 = False` flag to `True.
`table_5_output_without_imsrg2.txt` shows the example output for this script without IMSRG(2) results.

`figure_5_CCSD` computes the CCSD results presented in Fig. 5.
`figure_5_CCSD_output.txt` shows the example output for this script.
