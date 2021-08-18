from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = ['numpy', 'scipy']

setup(
    version="1.0",

    #find all packages using the following command.
    packages=["retire"],
    name='retire',
    scripts=[],
    
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
        # And include any *.msg files found in the 'hello' package, too:
        'hello': ['*.msg'],
    },

    author="Wenxin Zhou",
    author_email="wez243@ucsd.edu",
    description="Robustified Expectile Regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="retire",

    license='GPL-3.0',
    install_requires=install_requires,
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False
)