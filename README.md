# Celltype Prediction with Marioni Time-course Gastrulation Data

This repository documents the efforts to perform cell type prediction
on the
[MouseGastrulationData](https://github.com/MarioniLab/MouseGastrulationData)
single-cell RNA-seq timecourse data.

## Quick Start

For a quick dive into the data, the library, and the final prediction
score of 80%, a summarized notebook
[QuickStart.ipynb](./QuickStart.ipynb) is included to reproduce the
results of the analysis.


## Method

We pull E6.5 timepoint data of 2075 cells containing the raw gene
counts and phenotype samples, annotated with cell types as outlined by
*Marioni et al.* in their 2019 paper "A single-cell molecular map of
mouse gastrulation and early organogenesis".

### 1. Feature Selection

This is performed in the
[1_prime_the_data.ipynb](./1_prime_the_data.ipynb) notebook.

We pre-process the data using the
[Seurat](https://satijalab.org/seurat/) toolkit from the *Satija* lab,
stopping at the point where the top 2000 differentially expressed
genes are discovered by background modelling.

Additional prediction factors such as cell size (total transcripts) and cell
heterogeneity (total genes detected) are included as input variables.

This yields a training dataset of 2075 samples with 2002 features.

### 2. Initial Discovery and Library Conception

We perform initial discovery on the dataset in the
[2_celltype_prediction.ipynb](./2_celltype_prediction.ipynb) building
a rudimentary 2 latent layer neural network and assess the initial
training results.

Once it becomes clear that a parameter search will be needed to
optimize the prediction rate, much of the code is split out and
rewritten into a concise but flexible python library:

    CellTypePredictor.py

This library allows one to specify any number of layers with custom
input and output nodes, defaulting a linear fully connected dense
layer if unprompted. The ReLU rectifier is used in between all layers,
but can be overridden at construction, along with the Sigmoid output
layer activation function. The number of training epochs along with
batch sizes, and deterministic seeds can also be set.

### 3. Parameter Search

A wide parameter search consisting of varying layer characteristics,
epoch training ranges and batch sizes are tested, finally settling on
a configuration that consistently yields a prediction score of 80%.

The efforts are documented in the
[3_automated.ipynb](./3_automated.ipynb) notebook, and are then
summarized succintly in the [QuickStart.ipynb](./QuickStart.ipynb)
notebook.

