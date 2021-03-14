# P1-Analyze Trades

## Project Tracking

see _Project-Tracking.xlsx file

## Backend Overview

1. read in portfolio and trades from IB activity file (tradelog..py)
1. append own charactierstics
1. append market data
1. build models
1. decide best model
1. pickle file & show inputs via mlflow
1. validation exhibits
1. estimate predicted return for single record (see explainresults)

### MLFlow

1. Terminal ```pipenv sync``` to create environment, ```pipenv shell``` to enter environment
1. ```mlflow ui``` to enter environment
1. To shut down, type "ctrl + c" in terminal

## Frontend Build App

1. Start docker desktop app
1. Let's build our image: ```docker build -t analyze:latest . ```
1. and run: ```docker run -p 8004:8003 analyze:latest``` ,  host port : container port :
1. ```curl localhost:8004``` -> Hello world!
1. To turn off, ctrl+c in terminal

## Frontend Overview

1. read in csv file w own characteristics
1. append market data
1. build model, default, user-input model
1. run shap summary
1. allow individual inputs
1. show force plot
1. return model to user, if new
