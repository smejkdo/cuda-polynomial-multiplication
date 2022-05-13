# Polynomial multiplication accelerated by GPU

Implementation multiplies two polynomials using trivial algorithm in two different interpretations and also with Karatsuba algorithm, both on CPU and GPU. Output is time of calculation for different algorithms.

Folder data contains tables of computed times for different block sizes and sizes of input polynomials. Images contains graphs from this data and output of NVidia NVSight for different algorithms. Folder cpp contains CPU only implementation.

Program is run using "make run", changes to block size or size of polynomials has to be made in code.

