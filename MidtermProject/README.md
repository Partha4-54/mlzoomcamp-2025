Exoplanet Detection Using Machine Learning

Project Overview
This project detects exoplanets using machine learning by analyzing light curve data — the brightness of stars over time. When a planet passes in front of a star, the brightness briefly drops, known as a transit.
I use Kepler telescope data to build an ML model that can predict whether a star has an exoplanet based on its light curve.

Problem Statement
Can we use machine learning to automatically detect exoplanets
from light curve time-series data by analyzing transit dips?

Dataset Information
Column Type	Description
FLUX.1 – FLUX.3197	Light curve values (brightness over time)
LABEL	0 = No exoplanet, 1 = Exoplanet detected
Dataset source:
https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data
Place both files inside the data/ folder:
exoplanet_ml_project/data/exotrain.csv  
exoplanet_ml_project/data/exotest.csv

Future Improvements
Train 1D CNN for time-series pattern learning
Use AstroPy BLS method for orbital period detection
Deploy model via Streamlit / HuggingFace
Test on Kepler/TESS FITS telescope data�
