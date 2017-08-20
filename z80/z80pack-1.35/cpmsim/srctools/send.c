/*
 * Sends a file through named pipe "auxin" to the CP/M simulation
 *
 * Copyright (C) 1988-2017 by Udo Munk
 *
 * History:
 * 05-OKT-1988 Development on TARGON/35 with AT&T Unix System V.3
 * 11-MAR-1993 comments in english and ported to COHERENT 4.0
 * 01-OCT-2006 modified to compile on modern POSIX OS's
 * 09-MAR-2016 moved pipes to /tmp/.z80pack
 * 20-MAR-2017 renamed pipe
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>

void sendbuf(int);

char buf[BUFSIZ];
char cr = '\r';
int fdout, fdin;

int main(int argc,char *argv[])
{
	register int readn;

	if (argc != 2) {
		puts("usage: send filname &");
		exit(1);
	}
	if ((fdin = open(argv[1], O_RDONLY)) == -1) {
		perror(argv[1]);
		exit(1);
	}
	if ((fdout = open("/tmp/.z80pack/cpmsim.auxin", O_WRONLY)) == -1) {
		perror("pipe auxin");
		exit(1);
	}
	while ((readn = read(fdin, buf, BUFSIZ)) == BUFSIZ)
		sendbuf(BUFSIZ);
	if (readn)
		sendbuf(readn);
	close(fdin);
	close(fdout);
	return(0);
}

void sendbuf(int size)
{
	register char *s = buf;

	while (s - buf < size) {
		if (*s == '\n')
			write(fdout, (char *) &cr, 1);
		write(fdout, s++, 1);
	}
}
