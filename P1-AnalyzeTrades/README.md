# P1-Analyze Trades Architecture

## Purpose

Apply data science and analytics to a user's trading history to help user make better decisions
<https://analyze-trades-staging.herokuapp.com/>
<https://analyze-trades-prod.herokuapp.com/>
See ...Important\Startup\P1-AnalyzeTrades

## Architecture

C4 style approach, see README puml files and pngs

## Project Tracking

1. See [Azure DevOps, GitHub sign-in](https://dev.azure.com/swang4331/P1-AnalyzeTrades/_backlogs/backlog/P1-AnalyzeTrades%20Team/Features/?showParents=true)
1. Backup: see _Project-Tracking.xlsx file

## C2 Backend

### Local Scripts: Pnl Review

1. Go to Interactive Brokers -> activity statements -> pull standard .csv statement
1. Update raw `data` folder with account statement (`U1060261` in file name) and `pcm-tracking` loghist tab
1. Run standalone scripts a to e (just before building model)
1. Open `output\_pnl_review.xlsx`. This is the main file file for analysis
1. Paste `e_resultcleaned.csv` in `input_e_result` tab, making sure lookups at right are not broken

### Local Scripts: Model Update

1. enter virtual environment for P-1...
    1. if in vs-code can select environment from jupyter/python dropdown
1. read in portfolio and trades from IB activity statement (tradelog..py)
1. append own characteristics
1. append market data
1. build models (f_buildmodel), pickle file & show inputs via mlflow
1. validation exhibits (g_explainmodel)
1. test estimated predicted return for single record
1. **decide best model** by copying run from local mlflow to repo mlflow and update `app.py`
1. **update requirements in P1- folder** by running `pipenv lock --keep-outdated -d -r > ../requirements.txt` outside virtual env and copying to P1-... folder
    1. backup: pull requirements.txt directly from mlflow run artifacts
    1. ensure python version correct path `py -0p`, updating PATH variable order for system/user if want different default python version. System
    variables go first.
    1. can test full version name using `py -#.# --version`
    1. if updating python, need to manually remove virtual environment first
1. **run tests** including `pytest` and `flask run` and `docker build...` below
1. **save work as ipynb and html** by
    1. Saving jupyter interactive to original mlflow
    1. Copying run to repo
    1. and then use Anaconda -> JupyterLab -> save as HTML

### MLFlow & Virtual Env Update

1. First time build: Terminal `pipenv sync --dev` to install env locally with piplock or `pipenv update --dev` to update and install environment
1. Terminal: `pipenv shell` to enter environment or in vs code, right click open terminal in folder with pipfile
1. `mlflow ui --backend-store-uri file:D:/Stuff/OneDrive/MLflow` to enter environment (omit --backend if want to see test runs)
1. To shut down, type "ctrl + c" in terminal
1. Optional: `mlflow gc --backend-store-uri file:D:/Stuff/OneDrive/MLflow` to clean up deleted runs (e.g. deleted from mlflow ui)

### Build App Basic (Flask)

1. In pipenv virtual environment, `flask run`

## C2 Frontend Overview

see README_C2...png

### Build App Full Version

1. Start docker desktop app (outside any virtual environment)
1. Let's build our image: `docker build -t analyze:latest .`
1. and run: `docker run -p 8004:8003 analyze:latest` ,  host port : container port
1. `curl localhost:8004` -> website (does not match 127.0.0.0 diff from 0.0.0.0).
1. Test website directly via Docker Desktop links
1. To turn off, ctrl+c in terminal

### Single prediction

1. allow individual inputs
    1. later: append chars
1. use existing model
1. run shap summary
1. show force plot

### Batch version

1. read in csv file w own characteristics
1. use existing model
    1. later: consider building new model, default, user-input model
    1. append market data
1. run shap summary
1. allow individual inputs
1. show force plot
1. return model to user, if new

### Deployment notes

#### Heroku

1. Procfile
1. runtime.txt file to specify correct python version
1. requirements.txt

#### Docker (not used in prod)

1. heroku.yml
