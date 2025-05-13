from setuptools import setup, find_packages

setup(
        name="uteqipy", 
        version="0.0.3", 
        author="Keiichi Hashimoto", 
        author_email="khashimoto@eps.s.u-tokyo.ac.jp", 
        packages=find_packages(), 
        install_requires=[
            "matplotlib", 
            "numpy", 
            "dask", 
            "Netcdf4", 
            "pandas", 
            "xarray", 
            "pillow", 
            "scipy", 
            "scikit-image",
            ],
        )
