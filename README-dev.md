README for Development
==============================
# Install
First clone the repo 
```
git clone https://github.com/Labo-Lacourse/stepmix.git
``` 
Then in the project directory, install stepmix in editable mode. Creating a virtual environment is recommended, but
optional. Add the ```[dev]``` tag to install additional dev dependencies.
```
cd stepmix
python3 -m venv venv
source venv/bin/activate
pip install -e .[dev]
``` 
If you run into issues with editable installs, try ```pip install --upgrade pip```.

# Unit Testing
All tests can now be executed by running the following in the project directory.
```
pytest
``` 
Tests can be edited in the ```test``` directory. Please refer to the [pytest Documentation](https://docs.pytest.org/en/7.1.x/getting-started.html) for more
information on how to add tests. 

Moreover, the package is installed and tests are run after each commit using [Github Actions](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python) as configured by ```.github/workflows/pytest.yaml```.

# Documentation
Documentation is published using [Read the Docs](https://readthedocs.org/) and is available
[here](https://stepmix.readthedocs.io/en/latest/). The doc is automatically recompiled after each
commit to ```master``` branch.

Most of the existing Python docstring should be compiled automatically. Otherwise, the documentation can be edited by
changing files in ```docs/source```. You can compile the documentation locally by running
```
cd docs
make html
``` 
and open ```docs/build/html/index.html``` in a browser.

# Code  Formatting
StepMix uses the [Black](https://github.com/psf/black) formatting tool. Simply run
```
black .
``` 
in the project directory to reformat Python files.

# Bump versions
We use ```bumpver``` and the ```MAJOR.MINOR.PATCH``` convention. Use one of
```
bumpver update --patch 
bumpver update --minor 
bumpver update --major 
```
to update all files in the project. Add the ```--dry``` flag to 
see changes without committing them.

# Publish and update package
You need to be a maintainer to do this. We use [Flit](https://flit.pypa.io/en/stable/) for publishing the package. Make sure to set up
a ```.pypirc``` file in your ```HOME``` directory as described [here](https://flit.pypa.io/en/latest/upload.html). Following
changes to PyPI, you may need to set your username to ```__token__``` and your password to your PyPI API token.

You can test using  
```
flit publish --repository testpypi
``` 
then actually publish with
```
flit publish
``` 

