#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="idpfold",
    version="0.0.1",
    description="Precise Generation of Conformational Ensembles for Intrinsically Disordered Proteins Using Fine-tuned Diffusion Models",
    author="Junjie Zhu",
    author_email="shiroyuki@sjtu.edu.cn",
    url="https://github.com/Junjie-Zhu/IDPFold",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
            "preprocess_command = src.read_seqs:main"
        ]
    },
)
