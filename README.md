> SPIR Project 2024 Spring

This repo includes the code development for the SPIR project on IPCA factor portfolio optimization currently under development. The main branch includes the code for applying IPCA on the imputed kelly_data_without_nanocap.p data and applying QP for factor portfolio optimization. The imputation step can be found in IPCA.ipynb file. 
## Table of Contents

- [Installation](#installation)
- [File Structure](#files)

## Installation

My test envirionment can be found in environment.yml.

For IPCA, we use the python implementation of the Instrumtented Principal Components Analysis framework by Kelly, Pruitt, Su (2017), which can be found at https://github.com/bkelly-lab/ipca.

For quadratic programming, we use cvxopt.

## File Structure
```bash
ipca_factor_portfolio/
│
├── IPCA.ipynb              # Example notebook for running IPCA and QP optimization
├── ipca_utils.py           # Util functions for running IPCA
├── port_solver.py          # Quadratic Programming for factor portfolio optimization
├── run_job_nonreg.py       # Run job file without adding regularization terms
├── run_job_reg.py          # Run job file with regularization terms
├── logger.py               # Logger class for error tracking
├── environment.yml         # Conda environment file
├── .gitignore   
