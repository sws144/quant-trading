# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: p1analyzetrades
#     language: python
#     name: p1analyzetrades
# ---

# %% [markdown]
# # overall
#
#

# %% [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#overall" data-toc-modified-id="overall-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>overall</a></span><ul class="toc-item"><li><span><a href="#imports" data-toc-modified-id="imports-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>imports</a></span></li><li><span><a href="#run-notebooks" data-toc-modified-id="run-notebooks-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>run notebooks</a></span></li></ul></li></ul></div>

# %% [markdown]
# ## imports

# %%
import papermill as pm

# %%
notebooks = ["P1-AnalyzeTrades_a_tradelog.ipynb"]

# %% [markdown]
# ## run notebooks

# %%
for nb in notebooks:
    output_temp = nb.split('.')[0] + "_output." + nb.split('.')[1]
    pm.execute_notebook(nb, output_path= output_temp )


# %%
