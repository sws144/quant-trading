# P1-Analyze Trades Architecture

## Purpose

Apply data science and analytics to a user's trading history to help user make better decisions

## Architecture

C4 style approach, see README puml files and pngs

## Project Tracking

1. See [Azure DevOps, GitHub sign-in](https://dev.azure.com/swang4331/P1-AnalyzeTrades/_backlogs/backlog/P1-AnalyzeTrades%20Team/Features/?showParents=true)
1. Backup: see _Project-Tracking.xlsx file

## C2 Backend Overview

1. Update raw data `data` with account statement with `U1060261` in file name and `pcm-tracking` details
1. Run standalone scripts a to e (just before building model)
1. Pull in `data\_pnl_review.xlsx`
1. Paste in results tab, making sure lookups at right are not broken

### C3 Backend Outline

1. read in portfolio and trades from IB activity statement (tradelog..py)
1. append own characteristics
1. append market data
1. build models
1. decide best model
1. pickle file & show inputs via mlflow
1. validation exhibits
1. estimate predicted return for single record (see TODO explainresults)

### MLFlow

1. First time build: Terminal `pipenv sync` to create environment,
1. Terminal: `pipenv shell` to enter environment or in vs code, right click open terminal in folder with pipfile
1. `mlflow ui --backend-store-uri file:C:/Stuff/OneDrive/MLflow` to enter environment
1. To shut down, type "ctrl + c" in terminal
1. Optional: `mlflow gc --backend-store-uri file:C:/Stuff/OneDrive/MLflow` to clean up deleted runs (e.g. deleted from mlflow ui)

## C2 Frontend Overview

### Build App

1. Start docker desktop app
1. Let's build our image: `docker build -t analyze:latest .`
1. and run: `docker run -p 8004:8003 analyze:latest` ,  host port : container port
1. `curl localhost:8004` -> Hello world!
1. To turn off, ctrl+c in terminal

### single prediction

1. allow individual inputs
    1. later: append chars
1. use existing model
1. run shap summary
1. show force plot

### batch version

1. read in csv file w own characteristics
1. use existing model
    1. later: consider building new model, default, user-input model
    1. append market data
1. run shap summary
1. allow individual inputs
1. show force plot
1. return model to user, if new
