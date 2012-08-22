#ifndef LIBMISC_H
#define LIBMISC_H

#ifdef __cplusplus
extern "C" {
#endif

char *int2hex(int val,int digit);
char *int2asc(int val);
int htoi(char );
//int atoi(char *);
void trim(char *);
unsigned char htouc(char c);

char *get_token(char *src,int *p);
unsigned long hex_to_ulong(char *src);

void buzz_ok();
void buzz_ng();

unsigned long myrand();

#ifdef __cplusplus
}
#endif

#endif
