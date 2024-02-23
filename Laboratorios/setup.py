from setuptools import setup, find_packages

setup(
    name='unsupervised',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'matplotlib',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            # incluir comandos de consola aquí
        ],
    },
    author='Cesar Augusto Saenz Jimenez',
    author_email='cesar.saenz@udea.edu.co',
    description= "Este paquete de Modulos No Supervizado",
    long_description=open('README.md').read(),
    license=open('LICENSE.txt').read()
)
