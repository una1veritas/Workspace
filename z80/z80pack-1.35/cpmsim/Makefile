#Makefile
# Created on: 2017/08/19
#     Author: sin

all: cpmsim

cpmsim:
	cd ./srcsim 	&& \
	make -f Makefile.osx 	&& \
	make -f Makefile.osx clean
	
srctools:
	cd ./srctools 	&& \
	make 	&& \
	make install 	&& \
	make clean
