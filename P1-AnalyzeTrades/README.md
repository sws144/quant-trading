# P1-Analyze Trades Architecture

## Purpose

Apply data science and analytics to a user's trading history into order to drive value

## Architecture

C4 style approach, see README puml files and pngs

## Project Tracking

See [Azure DevOps](https://dev.azure.com/swang4331/P1-AnalyzeTrades/_backlogs/backlog/P1-AnalyzeTrades%20Team/Features/?showParents=true)
Backup: see _Project-Tracking.xlsx file

## Backend Overview

Can run as standalone scripts

1. read in portfolio and trades from IB activity statemetn (tradelog..py)
1. append own charactierstics
1. append market data
1. build models
1. decide best model
1. pickle file & show inputs via mlflow
1. validation exhibits
1. estimate predicted return for single record (see TODO explainresults)

### MLFlow

1. Terminal `pipenv sync` to create environment, `pipenv shell` to enter environment
1. `mlflow ui --backend-store-uri file:C:/Stuff/OneDrive/MLflow` to enter environment
1. To shut down, type "ctrl + c" in terminal

## Frontend Build App

1. Start docker desktop app
1. Let's build our image: `docker build -t analyze:latest .`
1. and run: `docker run -p 8004:8003 analyze:latest` ,  host port : container port
1. `curl localhost:8004` -> Hello world!
1. To turn off, ctrl+c in terminal

## Frontend Overview

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
