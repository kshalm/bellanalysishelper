from setuptools import setup

setup(name='analysishelper',
      version='0.1',
      description='Timetagger and analysis helper functions for the Bell Test',
      url='',
      author='Krister Shalm, Gautam Kavuri',
      author_email='lks@nist.gov',
      license='MIT',
      packages=['bellhelper', 'bellhelper.data'],
      install_requires=['pyyaml',
                        'bellMotors @ git+https://github.com/kshalm/motorLib.git#egg=bellMotors',
                        'zmqhelper @ git+https://github.com/kshalm/zmqhelpers.git#egg=zmqhelper',
                        'numpy',
                        'scipy',
                        'redis',
                        'numba'],
      include_package_data=True,
      zip_safe=False)
