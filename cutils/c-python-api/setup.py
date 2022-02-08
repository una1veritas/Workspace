from distutils.core import setup, Extension

module1 = Extension('hello',
                    sources = ['helloWrapper.c','hello.c'],
                    #extra_objects = ['hello.o'],
                    )

setup(name = 'hello', version = '1.0.0',  \
   ext_modules = [module1])