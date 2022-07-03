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
# # Overall
# See README.md for details
#

# %% [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Overall" data-toc-modified-id="Overall-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Overall</a></span><ul class="toc-item"><li><span><a href="#imports" data-toc-modified-id="imports-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>imports</a></span></li><li><span><a href="#run-notebooks" data-toc-modified-id="run-notebooks-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>run notebooks</a></span></li><li><span><a href="#build-models" data-toc-modified-id="build-models-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>build models</a></span></li></ul></li></ul></div>

# %% [markdown]
# ## imports

# %%
import papermill as pm

# %%
pnl_notebooks = [
    "P1-AnalyzeTrades_a_tradelog.ipynb",
    "P1-AnalyzeTrades_b_add_attr.ipynb",
    "P1-AnalyzeTrades_c_add_attr2.ipynb",
    "P1-AnalyzeTrades_d_exploredata.ipynb",
    "P1-AnalyzeTrades_e_feateng.ipynb",
]

# %% [markdown]
# ## run notebooks

# %%
for nb in pnl_notebooks:
    output_temp = nb.split('.')[0] + "_output." + nb.split('.')[1]
    _ = pm.execute_notebook(nb, output_path= output_temp )


# %% [markdown]
# ## build models

# %%
# typically run manually
# f notebooks and later
