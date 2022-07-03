pipenv update --dev
jupyter contrib nbextension install --user
jupyter nbextension enable toc2/main
jupyter nbextension enable varInspector/main
jupyter nbextension enable execute_time/ExecuteTime
echo y | jupyter kernelspec uninstall p1analyzetrades
python -m ipykernel install --user --name=p1analyzetrades