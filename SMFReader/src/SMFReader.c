/*
 ============================================================================
 Name        : SMFReader.c
 Author      : Sin
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

/* #include <unistd.h> */

typedef struct {
	uint16_t format, tracks, division;
} SMFInfo;

int SMF_checkheader(FILE * fp, SMFInfo * smf);

int main(const int argc, const char * argv[]) {
	puts("!!!Hello World!!!"); /* prints !!!Hello World!!! */

	if ( argc != 2 ) {
		return EXIT_FAILURE;
	}

	const char * smfname = argv[1];
	printf("name = %s\n", smfname);

	FILE * fp;
	SMFInfo smf;
	fp = fopen(smfname, "rb");
	if ( fp != NULL ) {

		if ( SMF_checkheader(fp, &smf) != 0 ) {
			printf("Seems SMF.\n");
			printf("format %d, tracks %d, division %d\n",smf.format, smf.tracks, smf.division);
		} else {
			printf("Not an SMF.\n");
		}
		fclose(fp);
	} else {
		printf("error on opening file.\n");
	}
	return EXIT_SUCCESS;
}

int SMF_checkheader(FILE * fp, SMFInfo * smf) {
	unsigned char tmp[6];
	rewind(fp);
	fread(tmp,1,4,fp);
	if (tmp[0] == 'M' && tmp[1] == 'T' && tmp[2] == 'h' && tmp[3] == 'd') {
		fread(tmp,1,4,fp);
		fread(tmp,1,6,fp);
		smf->format = tmp[0] << 8;
		smf->format += tmp[1];
		smf->tracks = tmp[2] << 8;
		smf->tracks += tmp[3];
		smf->division = tmp[4] << 8;
		smf->division += tmp[5];

		return 1;
	}
	return 0;
}
