/*
 ============================================================================
 Name        : ASCIIART_C.c
 Author      : Sin Shimozono
 Version     :
 Copyright   : GPL
 Description : Hello World in C, Ansi-style
 ============================================================================
 */
/* ASCIIART.BAS
10 FOR Y=-12 TO 12
20 FOR X=-39 TO 39
30 CA=X*0.0458
40 CB=Y*0.08333
50 A=CA
60 B=CB
70 FOR I=0 TO 15
80 T=A*A-B*B+CA
90 B=2*A*B+CB
100 A=T
110 IF (A*A+B*B)>4 THEN GOTO 200
120 NEXT I
130 PRINT " ";
140 GOTO 210
200 IF I>9 THEN I=I+7
205 PRINT CHR$(48+I);
210 NEXT X
220 PRINT
230 NEXT Y
 */

#include <stdio.h>
#include <stdlib.h>

int main(void) {
	puts("!!!Hello World!!!"); /* prints !!!Hello World!!! */

	for(int y=-12; y<= 12; ++y) {
		for(int x = -39; x <= 39; ++x) {
			float ca = x * 0.0458;
			float cb = y * 0.08333;
			float a = ca;
			float b = cb;
			int i = 0;
			while ( i < 16 ) {
				float t = a*a-b*b+ca;
				b = 2 * a * b + cb;
				a = t;
				if (a*a+b*b > 4)
					break;
				++i;
			}
			char c;
			if (i < 10 ) {
				c = '0' + i;
			} else if ( i < 16 ) {
				c = 'A' + i - 10;
			} else {
				// i == 16
				c = ' ';
			}
			putc(c, stdout);
		}
		puts("");
	}

	puts("Bye!");
	return EXIT_SUCCESS;
}
