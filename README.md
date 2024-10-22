 # copairs

`copairs` is a Python package for finding groups of profiles based on metadata and calculate mean Average Precision to assess intra- vs inter-group similarities.

## Getting started

### System requirements
copairs supports Python 3.8+ and should work with all modern operating systems (tested with MacOS 13.5, Ubuntu 18.04, Windows 10).

### Dependencies
copairs depends on widely used Python packages:
* numpy
* pandas
* tqdm
* statsmodels
* [optional] plotly

### Installation

To install copairs and dependencies, run:
```bash
pip install copairs
```

To also install dependencies for running examples, run:
```bash
pip install copairs[demo]
```

### Testing

To run tests, run:
```bash
pip install -e .[test]
pytest
```

## Usage

We provide examples demonstrating how to use copairs for:
- [grouping profiles based on their metadata](./examples/finding_pairs.ipynb)
- [calculating mAP to assess phenotypic activity and consistnecy of perturbation using real data](./examples/mAP_demo.ipynb)


## Citation
If you find this work useful for your research, please cite our [pre-print](https://doi.org/10.1101/2024.04.01.587631):

Kalinin, A.A., Arevalo, J., Vulliard, L., Serrano, E., Tsang, H., Bornholdt, M., Rajwa, B., Carpenter, A.E., Way, G.P. and Singh, S., 2024. A versatile information retrieval framework for evaluating profile strength and similarity. bioRxiv, pp.2024-04. doi:10.1101/2024.04.01.587631

BibTeX:
```
@article{kalinin2024versatile,
  title={A versatile information retrieval framework for evaluating profile strength and similarity},
  author={Kalinin, Alexandr A and Arevalo, John and Vulliard, Loan and Serrano, Erik and Tsang, Hillary and Bornholdt, Michael and Rajwa, Bartek and Carpenter, Anne E and Way, Gregory P and Singh, Shantanu},
  journal={bioRxiv},
  pages={2024--04},
  year={2024},
  doi={10.1101/2024.04.01.587631}
}
```
