# Prediction-Power
Simulation code of "Maximizing the Value of Predictions in Control: Accuracy Is Not Enough."

## Installation instructions

1. Install Python dependencies. We recommend using [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main). Setup the conda environment by running
   ```bash
   conda env update --file env.yml --prune
   ```

2. Ensure that LaTeX is installed. If not, run

   ```bash
   sudo apt update
   sudo apt install texlive texlive-latex-extra texlive-fonts-extra dvipng cm-super
   pip install latex
   ```

## Generate figures

Run
```bash
bash run_prediction_power_exp.sh
```
