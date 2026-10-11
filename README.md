# California Housing Predictor

This project predicts median house values for California using a machine learning regression model built with scikit-learn and served through a Flask web app.

## Project Overview

- Model type: Linear Regression
- Dataset: California Housing dataset from scikit-learn
- Goal: Estimate house value based on housing and geographic features
- Web app: Flask-based form for user input and prediction output

## Features Used

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude

## Setup

1. Open the project folder.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

## Run the App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5001/
```

## Training the Model

To train a new model and save it as `model.pkl`:

```bash
python train_model.py
```

## Files in This Project

- `app.py` — Flask app and prediction interface
- `train_model.py` — model training script
- `model.pkl` — trained machine learning model
- `requirement.txt` — Python dependencies

## Useful Links

1. [GitHub Account](https://github.com)
2. [Heroku Account](https://heroku.com)
3. [VS Code IDE](https://code.visualstudio.com/)
