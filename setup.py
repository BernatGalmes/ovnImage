from setuptools import setup

setup(name='ovnImage',
      version='0.3.2',
      description='Useful helper functions of image processing and machine learning',
      url='',
      author="Miquel Miró Nicolau, Bernat Galmés Rubert, Dr. Gabriel Moyà Alcover",
      author_email='miquelca32@gmail.com, bernat_galmes@hotmail.com, gabriel_moya@uib.es',
      license='MIT',
      packages=['ovnImage', 'ovnImage.plots'],
      install_requires=[
          'scikit-learn',
          'matplotlib',
          'seaborn',
          'pandas',
          'seaborn',
          'numpy',
          'scikit-image',
          'opencv-python'
      ],
      zip_safe=False)
