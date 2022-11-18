from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "event-analysis",
    version = "0.0.1",
    description = "This package allows you to run Event Coincidence Analysis and Event Synchronization on your event series on the CPU and Nvidia-GPU",
    py_modules = [ "EventAnalysis" ],
    package_dir = { '' : "EventAnalysis" },
    package_data = { '' : [ 'EventAnalysis/kernels/*.cu' ] } ,
    include_package_data=True,
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3"
    ],
    long_description = long_description,
    long_description_content_type = "text/markdown",
    install_requires = [ 
        "numba ~= 0.56.2",
        "numpy ~= 1.23.3",
        "pandas ~= 1.5.0"
    ],
    url = "https://github.com/professorcode1/Event-Analysis",
    author = "Raghav Kumar",
    author_email = "raghkum2000@gmail.com"

)