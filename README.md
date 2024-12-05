# Assignment Instructions

### Setup: Conda

Use conda to manage your python environments. You can find all the documentation and installation guides here. https://conda.io/projects/conda/en/latest/user-guide/getting-started.html

**Note for Windows users**: After installing Anaconda, a new shell called "Anaconda Prompt" will be installed on your machine. Use this shell instead of PowerShell to run all the commands that follow.

Once you have conda installed, simply create a conda environment.

    $ conda create --name taichi_env python=3.10.14

After installation, you can activate your environment

    $ conda activate taichi_env

And finally, install the requirements (this command should be run from the folder which contains this README file).

    $ pip install -e .

You can then run this assignment 

    $ python ./interactive/A1.py

### How to Submit

Simply gather all the python files within the **taichi_tracer** file (so everything EXCEPT the scene_data_dir) and compress them together into a zip file. name your zip_file <McGill_ID>.zip 

**DO NOT ADD ANYTHING BEFORE OR AFTER THE MCGILL ID.**

If your McGill ID is 123456789, then submit EXACTLY:

    123456789.zip
# ecse546-A4
