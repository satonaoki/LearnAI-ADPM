# This script creates a Python egg with the solutions for the hands-on
# labs, which can be installed as a custom library in Azure Databricks
#
# To updated the egg, run the following script:
# /usr/bin/python3 create_solution_egg.py bdist_egg


from setuptools import setup, find_packages
setup(
    name = "adpm_solutions",
    version = "0.1", # we are not performing actual version tracking at this point
    packages = find_packages() # looks for all packages under the current folder
)

