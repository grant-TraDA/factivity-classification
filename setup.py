import codecs
from setuptools import setup


with codecs.open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name = "Polish NLI",
    version = "0.0.1",
    author = "Karolina Seweryn, Anna Wróblewska, Daniel Ziembicki",
    author_email = "karolina.seweryn@pw.edu.pl",
    description = ("Polish NLI"),
    license = "MIT",
    url = "https://github.com/grant-TraDA/factivity-classification",
    install_requires=REQUIREMENTS,
    packages=['src']
)