# Climate Change Trend Analysis – India

## 1. Introduction

This project presents a prototype for analyzing climate change trends in India using data science techniques.

The project focuses on temperature, rainfall, and CO₂ emissions data for the period 2000–2024. It performs data preprocessing, validation, descriptive statistical analysis, correlation analysis, trend analysis, linear regression, and data visualization.

The project is aligned with **SDG 13 – Climate Action**.

---

## 2. Research Problem

### Climate Change Trend Analysis Using Data Science – India

The objective is to develop a data science prototype that analyzes climate-related trends in India using historical data from 2000 to 2024.

The prototype studies:

- Temperature trends
- Rainfall trends
- CO₂ emission trends
- Relationships between climate variables
- Temperature prediction using linear regression
- Statistical measures and visualizations

---

## 3. Objectives

The main objectives of this project are:

1. To develop a working prototype for climate data analysis.
2. To load and validate climate data.
3. To preprocess the dataset and handle invalid or missing values.
4. To calculate descriptive statistics.
5. To analyze correlations between climate variables.
6. To identify temperature, rainfall, and CO₂ emission trends.
7. To apply linear regression for temperature analysis.
8. To generate graphical visualizations.
9. To use Git and GitHub for version control.
10. To make the project reproducible through documentation and testing.

---

## 4. Technologies and Tools Used

- Python
- VS Code
- Git
- GitHub
- pandas
- NumPy
- Matplotlib
- scikit-learn
- pytest
- Markdown

---

## 5. Dataset

The project uses a CSV dataset containing yearly climate observations from **2000 to 2024**.

The dataset used in the current prototype contains the following columns:

| Column | Description |
|---|---|
| Year | Year of observation |
| Temperature_C | Temperature value in Celsius |
| Rainfall_mm | Rainfall value in millimetres |
| CO2_Emissions_tonnes | CO₂ emissions in tonnes |

The input file is:

```text
data/india_climate_2000_2024.csv