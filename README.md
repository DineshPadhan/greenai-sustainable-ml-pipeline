# Green AI Pipeline - Kaggle Community Olympiad: Hack 4 Earth

![Green AI](https://img.shields.io/badge/Green-AI-success)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue)

## 🌍 Project Overview

This project demonstrates sustainable machine learning practices by optimizing neural networks for minimal environmental impact while maintaining model performance. We implement carbon-aware computing, neural architecture search (NAS), and model compression techniques to reduce the carbon footprint of AI systems.

**Competition**: [Kaggle Community Olympiad: Hack 4 Earth - Green AI Challenge](https://www.kaggle.com/code/dineshpadhan2023/green-ai)

## 🎯 Key Features

- **Carbon Emission Tracking**: Real-time monitoring using CodeCarbon
- **Neural Architecture Search (NAS)**: Automated optimization for efficiency
- **Model Compression**: Reduced model size without significant performance loss
- **Carbon-Aware Computing**: Region and time-based optimization
- **Comprehensive Footprint Documentation**: Detailed evidence and methodology

## 📊 Results Summary

| Metric | Baseline Model | Optimized Model | Improvement |
|--------|---------------|-----------------|-------------|
| Model Size | ~0.5 MB | ~0.3 MB | 40% reduction |
| Parameters | 32K+ | 16K-24K | 25-50% reduction |
| Carbon Emissions | Variable | Reduced | 15-45% reduction |
| Performance | Baseline | Maintained/Improved | ≥95% baseline |

## 🚀 Quick Start

### ✅ Tested Platforms

This methodology has been **validated and tested** on:

- ✅ **Google Colab** (T4 GPU) - Fastest training
- ✅ **Kaggle Notebooks** (CPU) - Most energy efficient
- ✅ **Local Environment** (CPU/GPU) - Full control

**Recommendation**: Use Colab for development, Kaggle for final result.

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
pandas, numpy, matplotlib, seaborn
codecarbon (for emission tracking)
optuna (optional, for advanced NAS)
```

### Installation

#### Option A: Google Colab (Recommended for Quick Testing)

1. Open this notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)] and Add competition data

2. Enable T4 GPU: `Runtime → Change runtime type → T4 GPU`

3. Run all cells - Training completes in ~3 minutes!

#### Option B: Kaggle Notebooks (Recommended for Submission)

1. Fork this kernel: https://www.kaggle.com/code/dineshpadhan2023/green-ai

2. Add competition data automatically

3. Run and submit directly from Kaggle

#### Option C: Local Installation

1. **Clone the repository**:
```bash
git clone https://github.com/DineshPadhan/greenai-sustainable-ml-pipeline
cd greenai-sustainable-ml-pipeline
```

2. **Install dependencies**:
```bash
pip install tensorflow pandas numpy matplotlib seaborn codecarbon optuna scikit-learn
```

3. **Download competition data**:
   - Visit [Kaggle Competition Page](https://www.kaggle.com/competitions/kaggle-community-olympiad-hack-4-earth-green-ai)
   - Download `train.csv`, `test.csv`, and `metaData.csv`
   - Place files in the project root directory

### Running the Pipeline

#### Jupyter Notebook
```bash
jupyter notebook green_ai_pipeline_final.ipynb
```
Run all cells sequentially to execute the complete pipeline.

### Expected Outputs

The pipeline generates the following files in the `outputs/` directory:

- `optimal_model_[timestamp].keras` - Best performing model
- `baseline_model_[timestamp].keras` - Baseline comparison model
- `sample_submission.csv` - Kaggle submission file
- `emissions.csv` - CodeCarbon emission logs
- Various visualization plots

## 🔬 Methodology

### 1. Carbon Emission Tracking
- **Tool**: CodeCarbon v2.x
- **Metrics**: kWh, kg CO2e, runtime
- **Scope**: Training, NAS, inference

### 2. Neural Architecture Search
- **Framework**: Optuna (with grid search fallback)
- **Objective**: Minimize validation loss while reducing parameters
- **Search Space**: Layer count, units, dropout, learning rate

### 3. Model Compression
- **Techniques**: Parameter reduction, efficient architectures
- **Target**: 40-60% size reduction with <5% performance loss

### 4. Carbon-Aware Computing
- **Analysis**: Regional carbon intensity from metadata
- **Strategy**: Prefer low-carbon regions and time windows

## 📈 Data Sources & Licenses

### Competition Data
- **Source**: Kaggle Community Olympiad
- **License**: Competition rules apply
- **Files**: `train.csv`, `test.csv`, `metaData.csv`
- **URL**: https://www.kaggle.com/competitions/kaggle-community-olympiad-hack-4-earth-green-ai

### Carbon Intensity Data
- **Tool**: CodeCarbon embedded grid factors
- **Source**: Electricity Maps API / regional grids
- **License**: CodeCarbon MIT License

### Software Libraries
All libraries used are open-source:
- TensorFlow: Apache 2.0
- scikit-learn: BSD 3-Clause
- Pandas, NumPy: BSD 3-Clause
- CodeCarbon: MIT License
- Optuna: MIT License

## 📝 Documentation

- **`FOOTPRINT.md`**: Detailed carbon footprint measurement methodology
- **`evidence.csv`**: Row-by-row emission data for baseline and optimized runs
- **`data_card.md`**: Dataset documentation and preprocessing details
- **`model_card.md`**: Model architecture, performance, and limitations
- **`carbon_aware_decision.json`**: Carbon-aware scheduling decisions

## 🤝 Contributing

This is a competition submission repository. For educational purposes:
1. Fork the repository
2. Experiment with different optimization strategies
3. Share your results and insights

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Competition Submission

- **Kaggle Notebook**: https://www.kaggle.com/code/dineshpadhan2023/green-ai
- **Submission File**: `outputs/sample_submission.csv`
- **Format**: CSV with columns `Id` and `GreenScore`

## 👨‍💻 Author

**Dinesh Padhan**
- Kaggle: [@dineshpadhan2023](https://www.kaggle.com/dineshpadhan2023)
- Competition: Kaggle Community Olympiad - Hack 4 Earth

## 🌟 Acknowledgments

- Kaggle Community Olympiad organizers
- CodeCarbon team for emission tracking tools
- TensorFlow and Optuna communities
- Open-source AI/ML community

## 📞 Contact & Support

For questions or issues:
- Open an issue on GitHub
- Contact via Kaggle profile
- Refer to competition discussion forum

---

**🌱 Green AI Matters**: This project demonstrates that we can build powerful AI while being responsible stewards of our planet's resources.

**Last Updated**: October 30, 2025
