all: bas2img bin2wav wav2bin

bas2img: bas2img_bas2img_600.c 
	 gcc -Wall -o bas2img bas2img_600.c
bin2wav: bin2wav_200b.c
	 gcc -Wall -o bin2wav bin2wav_200b.c
wav2bin: wav2bin_200a.c
	 gcc -Wall -o wav2bin wav2bin_200a.c
	 
clean: 
	 rm bas2img bin2wav wav2bin