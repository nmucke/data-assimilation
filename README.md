[![Build Status](https://github.com/jfhbuist/hello-world-package/actions/workflows/CI.yml/badge.svg?event=push)](https://github.com/jfhbuist/hello-world-package/actions)
[![codecov](https://codecov.io/gh/jfhbuist/hello-world-package/branch/master/graph/badge.svg?token=C4OJDHTMWJ)](https://codecov.io/gh/jfhbuist/hello-world-package)

# hello-world-package

This is a simple python package template.  
It uses pip for installation, flake8 for linting, pytest for testing, and coverage for monitoring test coverage.

To use it, first create a virtual environment, and install flake8, pytest, and coverage using pip.  
The following works on Windows: 
```
py -3 -m venv .venv
.venv\scripts\activate
python -m pip install --upgrade pip
pip install flake8 pytest coverage
```

Then, install the package, run it, and test it:
```
pip install -e .
python -m hello_world_package
flake8
coverage run -m pytest
coverage report
```

If not developing, but only using the package, just do:
```
pip install .
python -m hello_world_package
```

The package will now be listed when running:
```
pip freeze
```

It can be uninstalled using:
```
pip uninstall hello_world_package
```

Deactivate virtual environment:
```
deactivate
```
