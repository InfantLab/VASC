# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Video Actor Synchroncy and Causality (VASC)
# ## RAEng: Measuring Responsive Caregiving Project
# ### Caspar Addyman, 2020
# ### https://github.com/infantlab/VASC
#
# # Step 0 - Before we start
#
# Before anything else we need to install a few apps and libraries to make everything work. You should only have to do these steps once when you first install everything.

# ### 0.0 - Jupyter notebook environment
#
# *If you can read this you are probably already running Jupyter. Congratulations!*
#
# We recommend using the Anaconda Data Science platform (Python 3 version) 
#
# https://www.anaconda.com/distribution/
#

# ### 0.1 - OpenPoseDemo application
#
# Next we need to download and install the [OpenPoseDemo](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/demo_overview.md) executable.
#
# Additionally, you need to download the trained neural-network models that OpenPose uses. To do this go to the `models` subdirectory of OpenPose directory, and double-click / run the `models.bat` script.
#
# The `openposedemo` bin/exe file can be run manually from the command line. It is worth trying this first so you understand what  `openposedemo` is. See [this guide](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/demo_overview.md) or open a terminal app or Windows Powershell, navigate to the openpose installation folder and then try this command
#
# ```
# :: Windows
# bin\OpenPoseDemo.exe --video examples\media\video.avi --write_json output
# # Mac/Linux
# ./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_json output
# ```

# ### 0.2 - Load python libraries
#
# There are a handful of python libraries that we use for things like image manipulation, file operations, maths and stats. Many are probably already installed by default such as `os, math, numpy, pandas, matplotlib`. Others need adding to our python environment. 
#
# PyArrow is a useful extension for saving Pandas and NumPy data. We need it to move the large array created in Step 2 to Step 3. 
#
# **If you are using conda then run the following command to install all the main libraries.**
# ```
# conda install glob2 opencv pyarrow xlrd openpyxl
# ```
# #### Troubleshooting
# If when you run the code in Steps 1, 2 & 3 you might see an error like `ModuleNotFoundError: No module named 'glob'` this is because that python module needs to be installed on your computer. If you use Anaconda, the missing module can usually be installed with the command `conda install glob`.

# ### 0.3 - iPyWidgets
#
# We also install `iPyWidgets` & `iPyCanvas` in order to use buttons and sliders and images in Jupyter notebooks (these are used in Step 2). 
# ```
# conda install -c conda-forge ipywidgets 
# conda install -c conda-forge ipycanvas
# ```
# To make these work with the newer Jupyter Lab we also need to install the widgets lab extension, like so:
#
# ```
# Jupyter 3.0 (current)
# conda install -c conda-forge jupyterlab_widgets
#
# Jupyter 2.0 (older)
# conda install -c conda-forge nodejs
# jupyter labextension install @jupyter-widgets/jupyterlab-manager
# jupyter labextension install @jupyter-widgets/jupyterlab-manager ipycanvas
# ```
#
# Documentation:
# + https://ipywidgets.readthedocs.io/en/latest/
# + https://ipycanvas.readthedocs.io/en/latest/
# + https://ipycanvas.readthedocs.io/en/latest/installation.html#jupyterlab-extension
#

# ### 0.4 - JupyText 
#
# The standard ipython notebook format (`mynotebook.ipynb`) is a single file that mixes together code, formatting commands and outputs both as the results of running code and embedded binaries (images, graphs). This makes it non-human readable and very hard to tell what changes from one improvement to the next. `Jupytext` solves this by creating a synchronised plain text version of the file saved as a plain `.py` file (`mynotebook.py`). These are useful for developers (as it helps you track differences between versions more easily) but can mostly be ignored by users. 
#
#
# Install JupyText by running
#
# ```conda install -c conda-forge jupytext```

# ### 0.6 Notebook extensions (optional)
# Installing Jupyter notebook extenstions provide some useful tools for navigating notebooks (e.g. table of contents) and other features.
#
# To install, run these commands in terminal window.
#
# ```conda install -c conda-forge jupyter_nbextensions_configurator```
#
# ```jupyter nbextensions_configurator enable --user```
#
# * https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/index.html
# * https://moonbooks.org/Articles/How-to-create-a-table-of-contents-in-a-jupyter-notebook-/
#

# ### 0.6 - Using Jupyter with network drives
#
# By default Jupyter launched from Anaconda Navigator will open it in your home directory. It then might not be possible to access files on a network drive you need. To get around this first launch a command window for the correct Jupyter environment. Then use this command to launch Jupyter itself (assuming you want to access the U:/ drive). 
#
# ```jupyter lab --notebook-dir U:/```
