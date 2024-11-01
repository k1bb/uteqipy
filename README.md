# uteqipy  
**uteqipy** is a Python library for analysing images of precipitation particles taken with Rainscope (Suzuki et al., 2023).  
It currently suppoorts image cleaning, binarisation and elliptical fitting of liquid raindrops.  
## How to use
```python
import uteqipy as up

factory = "20240719-0430Z"
f = up.Factory(factory)
# Your working directory is created as "./{factory}".

f.read_frames("your/jpg/dir/*.jpg")
# All images are loaded and saved as a single netcdf file "./{factory}/{factory}_original.nc".

f.clean_frames()
# The data with the background brightness subtracted is saved in "./{factory}/{factory}_cleaned.nc".

f.binarize_frames()
# Binarized data is saved in "./{factory}/{factory}_binarized.nc".

f.label_frames()
# The results of labelling each object are stored in "./{factory}/{factory}_labeled.nc".

f.fit_objects()
# The results of elliptical fitting are stored in "./{factory}/{factory}_fitted.csv".

f.generate_all_reports()
# Analysis reports are generated in "./{factory}/reports".
```
