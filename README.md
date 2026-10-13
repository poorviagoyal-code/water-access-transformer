# # Transformer-Based Forecasting of Drinking-Water Access

A Data Science and Deep Learning project that uses a Transformer Encoder to predict country-level drinking-water access using historical data and socioeconomic indicators.

## Overview

Access to drinking water is an important public health and development indicator. This project explores whether a Transformer-based deep learning model can forecast water-access values using historical observations, urban population, and unemployment data.

The Transformer is compared with two baseline machine learning models: Linear Regression and Random Forest.

## Objectives

- Analyze historical drinking-water access data across countries.
- Explore relationships between water access, urban population, and unemployment.
- Build a Transformer Encoder for time-series forecasting.
- Compare the Transformer with traditional machine learning models.
- Evaluate predictions using MAE, RMSE, and R².

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- PyTorch
- Google Colab

## Datasets

The project uses three datasets:

1. **Water access:** WHO/UNICEF Joint Monitoring Programme (JMP) data.
2. **Urban population:** World Bank indicator — [Urban population (% of total population)](https://data.worldbank.org/indicator/SP.URB.TOTL.IN.ZS).
3. **Unemployment:** World Bank indicator — [Unemployment, total (% of total labor force)](https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS).

Water data source: [WHO/UNICEF JMP](https://washdata.org/data).

The raw CSV files are not included in this repository. Obtain them from their sources and check their applicable terms of use.

## Methodology

1. **Data preprocessing:** Clean the water-access data, standardize column names, reshape the World Bank datasets, and merge records by country code and year.
2. **Exploratory data analysis:** Examine the distribution of water access, yearly trends, and correlations between variables.
3. **Sequence creation:** Use five consecutive years of observations, with three features per year: water access, urban population, and unemployment.
4. **Data splitting:** Use target years through 2018 for training, 2019–2020 for validation, and 2021–2024 for testing.
5. **Feature scaling:** Fit standardization scalers using training data only.
6. **Transformer training:** Train a two-layer Transformer Encoder with four attention heads and use validation loss for early stopping.
7. **Model evaluation:** Compare Linear Regression, Random Forest, and Transformer predictions.

## Model Evaluation

The submitted experiment produced the following test-set results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.0867 | 0.2446 | 0.99993 |
| Random Forest | 0.4012 | 0.6368 | 0.99953 |
| Transformer | 0.3651 | 0.5117 | 0.99969 |

Linear Regression achieved the lowest MAE and RMSE in this experiment. Therefore, the Transformer did not outperform both baseline models on these error metrics.

The unusually high R² values should be interpreted cautiously, and further validation is needed. At least one Transformer prediction exceeds 100%, which is a limitation because the target is treated as a percentage.

## Repository Structure

```text
water-access-transformer/
├── deep_learning.py
├── model/
│   └── water_transformer.pt
├── results/
│   ├── model_comparison.csv
│   └── water_predictions.csv
├── report/
│   └── project_report_results.docx
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

The code was exported from Google Colab and expects the input CSV files in `/content`.

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `deep_learning.py` or copy its code into notebook cells.
3. Upload `water.csv`, `urban_population.csv`, and `unemployment.csv` to `/content`.
4. Run the notebook from top to bottom.
5. Review the generated plots, metrics, and prediction files.

Install the required libraries if needed:

```bash
pip install pandas numpy matplotlib scikit-learn torch
```

The Python file currently contains Colab-specific commands, so running it directly as a standalone Python script may require modifications.

## Output Files

- `model_comparison.csv` — evaluation metrics for the three models.
- `water_predictions.csv` — actual and predicted water-access values for test observations.
- `water_transformer.pt` — trained PyTorch checkpoint containing model parameters and preprocessing statistics.

The `.pt` file is intended to be loaded using PyTorch; it is not a document that can be opened in a browser.

## Limitations

- Results depend on data availability, missing observations, and country coverage.
- The current evaluation is rolling one-step-ahead prediction for target years 2021–2024, not a direct long-range forecast to 2030.
- The target's exact water-service category must be verified in the source data before describing it specifically as safely managed drinking-water access.
- Predictive associations do not establish causal relationships.
- Further analysis is needed to investigate unusually high R² values and predictions outside the 0–100% range.

## Future Improvements

- Compare a water-access-only model with the model using socioeconomic indicators.
- Evaluate prediction errors by country and year.
- Investigate methods that keep predictions within the valid percentage range.
- Test additional forecasting models and validation strategies.
- Explore longer-horizon forecasting with a clearly defined evaluation procedure.

## Author

**Poorvi Goyal**

Data Science | Deep Learning | Time-Series Forecasting

## Acknowledgements

Data sources: WHO/UNICEF Joint Monitoring Programme and the World Bank.
