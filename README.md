# MAMBAQUANT: QUANTIZING THE MAMBA FAMILY WITH VARIANCE ALIGNED ROTATION METHODS

![MAMBAQUANT Logo](https://github.com/yourusername/mambaquant/blob/main/assets/logo.png)

## Overview

**MAMBAQUANT**, a novel post-training quantization framework tailored for Mamba models, addressing key challenges such as outliers and variance inconsistencies. MambaQuant achieves less than 1% accuracy loss in quantizing weights and activations to 8-bit for various Mamba-based tasks, marking the first comprehensive PTQ design for this family.

This repository accompanies our ICLR 2025 paper titled **"MAMBAQUANT: QUANTIZING THE MAMBA FAMILY WITH VARIANCE ALIGNED ROTATION METHODS"**. 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Features

- **KLT-Enhanced Rotation**: 
- **Smooth-Fused Rotation**: 
- **Performance Optimization**: 

## Installation

Python 3.10.13  
◦ conda create -n your_env_name python=3.10.13  
torch 2.1.1 + cu118  
◦ pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118  
Requirements: vim_requirements.txt  
◦ pip install -r vim/vim_requirements.txt   
Install causal_conv1d and mamba  
◦ pip install -e causal_conv1d>=1.1.0  
◦ pip install -e mamba-1p1p1

## Usage
The code is being sorted out

## Contributing

We welcome contributions from the research and development community! Whether you're interested in improving the existing features, adding new functionalities, or reporting issues, your input is invaluable.

## Citation

If you find MAMBAQUANT useful in your research, please consider citing our paper:

```bibtex
@inproceedings{mambaquant2025,
  title={MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods},
  ...
}
