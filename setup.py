from setuptools import setup, find_packages

setup(name='analysishelper',
      version='0.1',
      description='Timetagger and analysis helper functions for the Bell Test',
      url='',
      author='Krister Shalm, Gautam Kavuri',
      author_email='lks@nist.gov',
      license='MIT',
      packages=find_packages(),
      install_requires=['pyyaml',
                        'zmqhelper @ git+https://github.com/kshalm/zmqhelpers.git',
                        'numpy',
                        'scipy',
                        'numba'],
      include_package_data=True,
      zip_safe=False)