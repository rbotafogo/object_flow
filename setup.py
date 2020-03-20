# from distutils.core import setup
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}    
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("object_flow.util.geom",
                  ["object_flow/util/geom.pyx"]),
        Extension("object_flow.util.bbox",
                  ["object_flow/util/bbox.pyx"]),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("object_flow.util.geom",
                  ["object_flow/util/geom.pyx"]),
        Extension("object_flow.util.bbox",
                  ["object_flow/util/bbox.pyx"]),
    ]


setup(
    name='object_flow',
    author='Rodrigo Botafogo',
    author_email='rodrigo.a.botafogo@gmail.com',
    cmdclass=cmdclass,
    install_requires=[
        'cmake',
        'dlib',
        'opencv-contrib-python',
        'imutils',
        'dictdiffer',
        'pytz',
        'scipy',
        'thespian'
    ],
    ext_modules=cythonize(ext_modules, language_level = "3"),
)
    
# setup(
#     ext_modules = cythonize("object_flow/util/geom.pyx")
#)
