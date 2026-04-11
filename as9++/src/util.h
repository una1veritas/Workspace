/*
 * util.h
 *
 *  Created on: 2026/04/07
 *      Author: sin
 */

#ifndef ASM09_UTIL_H_
#define ASM09_UTIL_H_

//#include <string.h>
//#include "as.h"

int any(const char, const char *);

void fatal(const char *str);
void print_line(void);
void f_record(void);
int delim(char c);
char * skip_white(char * ptr);
char mapdn(char c);
void error(const char * str);
int emit(int byte);
void eword(int wd);
int alpha(char);
int alphan(char);
int head(const char *str1, const char *str2);
void warning(const char *str);
//char * alloc(int nbytes);
void hexout(int byte);

int lobyte(int i);
int hibyte(int i);

#endif /* ASM09_UTIL_H_ */
