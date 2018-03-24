#!/usr/bin/env python

from setuptools import setup

ENTRY_POINTS = {
    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        'exampletutorials = weta.gui.tutorials',
    ),
    'orange.addons': (
        'Weta = weta.gui',
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    weta/gui/widgets/__init__.py
        'Weta Data = weta.gui.widgets.data',
        'Weta Preprocess = weta.gui.widgets.preprocess',
        'Weta Feature = weta.gui.widgets.feature',
        'Weta Learn = weta.gui.widgets.learn',
        'Weta Visualisation = weta.gui.widgets.visualise',
        'NZ Police = weta.gui.widgets.nzpolice',
        'Collo = weta.gui.widgets.collocation'
    ),

    # Register widget help
    "orange.canvas.help": (
        'html-index = weta.gui.widgets:WIDGET_HELP_PATH'
    )
}

NAMESPACES = ["weta"]
KEYWORDS = (
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3 add-ons',
    'Spark',
    'Spark ML',
    'Machine Learning'
)

LONG_DESCRIPTION = open('README.md').read()
LICENSE = open('LICENSE').read()

if __name__ == '__main__':
    setup(
        name="weta",
        version='0.0.1',
        author='Chao',
        author_email='richd.yang@gmail.com',
        url='https://github.com/richdyang/weta',
        description='A series of Widgets for Orange3 to work with Spark ML',
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        packages=['weta',
                  'weta.gui',
                  'weta.gui.utils',
                  'weta.gui.tutorials',
                  'weta.gui.widgets',
                  'weta.gui.widgets.data',
                  'weta.gui.widgets.preprocess',
                  'weta.gui.widgets.feature',
                  'weta.gui.widgets.learn',
                  'weta.gui.widgets.visualise',
                  'weta.gui.widgets.nzpolice',
                  'weta.gui.widgets.collocation'],
        package_data={
            'weta.gui': ['tutorials/*.ows'],
            'weta.gui.widgets': ['icons/*', 'assets/*'],
        },
        install_requires=[
            'Orange3',
            'pandas',
            'py4j',
            'sqlparse',
            'pyspark',
            'plotly'
        ],
        extras_require={
            'pyspark': [],

        },
        entry_points=ENTRY_POINTS,
        keywords=", ".join(KEYWORDS),
        namespace_packages=NAMESPACES,
        include_package_data=True,
        zip_safe=False,
    )
