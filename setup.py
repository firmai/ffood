from setuptools import setup

setup(name='ffood',
      version='0.1',
      description='Prediction model input analysis',
      url='https://github.com/firmai/ffood',
      author='snowde',
      author_email='d.snow@firmai.org',
      license='MIT',
      packages=['ffood'],
      install_requires=[
          'pandas',
          'numpy',
          'lightgbm',
          'pandas',
          'shap'
      ],
      zip_safe=False)
