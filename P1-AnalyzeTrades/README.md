# P1-Analyze Trades

## Overview

1. trade log to pnl calculator;
1. add other mkt features (multiple)
1. estimate predicted return

## Project Tracking

see _Project-Tracking.xlsx file

## MLFlow

1. Terminal ```pipenv sync``` to create environment, ```pipenv shell``` to enter environment
1. ```mlflow ui``` to enter environment
1. To shut down, type "ctrl + c" in terminal


## Build App
1. Start docker desktop app
1. Let's build our image: ```docker build -t analyze:latest . ```
1. and run: ```docker run -p 8004:8003 analyze:latest``` ,  host port : container port :
1. ```curl localhost:8004``` -> Hello world!
1. To turn off, ctrl+c in terminal