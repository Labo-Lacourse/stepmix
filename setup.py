from setuptools import setup, find_packages

setup(name='lca',
      version='0.0',
      python_requires='>=3.8',
      description='main package',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "pandas",
          "scikit-learn",
          "scipy",
      ])
