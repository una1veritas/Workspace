/*
 ============================================================================
 Name        : TestCode.c
 Author      : Sin
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct DATA {
	unsigned short id;
	char name[16];
};

int greaterThanID(struct DATA * a, struct DATA * b) {
	if ( a->id > b->id )
		return 1;
	return 0;
}

int greaterThan(struct DATA * a, struct DATA * b) {
	if ( strcmp(a->name, b->name) > 0 )
		return 1;
	return 0;
}

void swap(struct DATA * a, struct DATA * b) {
	struct DATA tmp;
	tmp = *a;
	*a = *b;
	*b = tmp;
	return;
}

void sort(struct DATA a[], int begin, int end) {
	int i, j;
	for(i = begin; i < end; i++) {
		for(j = i + 1; j < end; j++) {
			if ( greaterThan(&a[i], &a[j]) == 1 ) {
				swap(&a[i], &a[j]);
			}
		}
	}
}

int main(int argc, char * argv[]) {
	struct DATA data[16];
	struct DATA tmp;
	int count, i, tid;

	puts("!!!Hello World!!!"); /* prints !!!Hello World!!! */

	printf("argc = %d\n", argc);
	count = 0;
	for(i = 1; i < argc - 1; i += 2) {
		tid = atoi(argv[i]);
		if ( tid == 0 ) break;
		tmp.id = tid;
		if ( strlen(argv[i+1]) > 15 ) {
			strncpy(tmp.name, argv[i+1], 15);
			tmp.name[15] = 0;
		} else {
			strcpy(tmp.name, argv[i+1]);
		}
		data[count++] = tmp;
		printf("(%d: %s), \n", tmp.id, tmp.name);
	}
	puts("\nReading arguments finished.\n");

	for(i = 0; i < count; i++) {
		printf("id: %d, name: %s\n", data[i].id, data[i].name);
	}
	puts("");

	sort(data, 0, count);
	puts("After sorting:");
	for(i = 0; i < count; i++) {
		printf("id: %d, name: %s\n", data[i].id, data[i].name);
	}

	return EXIT_SUCCESS;
}
