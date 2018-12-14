from setuptools import setup

setup(name='ovnImage',
      version='0.2',
      description='Useful helper functions of image processing and machine learning',
      url='',
      author="Miquel Miró Nicolau, Bernat Galmés Rubert, Dr. Gabriel Moyà Alcover",
      author_email='miquelca32@gmail.com, bernat_galmes@hotmail.com, gabriel_moya@uib.es',
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
