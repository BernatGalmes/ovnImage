from setuptools import setup

setup(name='ovnImage',
      version='0.1',
      description='Functions for pca',
      url='',
      author='UIB-UGIVIA',
      author_email=['miquelca32@gmail.com', 'bernat_galmes@hotmail.com', 'gabriel_moya@uib.es'],
      license='MIT',
      packages=['ovnImage'],
      install_requires=[
          'scikit-learn',
          'matplotlib',
          'numpy',
          'seaborn',
          'scikit-image',
      ],
      zip_safe=False)
