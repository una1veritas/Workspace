
SRC = ./Sources

all: bas2img.exe bin2wav.exe wav2bin.exe

bas2img.exe: $(SRC)/bas2img_600.c 
	 gcc -Wall -o bas2img.exe $(SRC)/bas2img_600.c
bin2wav.exe: $(SRC)/bin2wav_200b.c
	 gcc -Wall -o bin2wav.exe $(SRC)/bin2wav_200b.c
wav2bin.exe: $(SRC)/wav2bin_200a.c
	 gcc -Wall -o wav2bin.exe $(SRC)/wav2bin_200a.c

clean: 
	 rm bas2img.exe bin2wav.exe wav2bin.exe
