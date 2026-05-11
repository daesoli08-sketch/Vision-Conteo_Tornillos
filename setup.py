from setuptools import setup, find_packages

setup(
    name="vision_conteo",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
    ],
    author="Equipo 4",
    description="Libreria para deteccion y conteo de tuercas y tornillos",
)