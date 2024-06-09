felica_readwrite: felica_readwrite.cpp
	cc -c -DUNICODE HardwareSerial.cpp
	cc -c -DUNICODE RCS620S.cpp
	cc  -lssl -lcrypto -lstdc++ -lwiringPi RCS620S.o HardwareSerial.o -DUNICODE felica_readwrite.cpp -o felica_readwrite
