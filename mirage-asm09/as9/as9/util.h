/*
 * util.h
 *
 *  Created on: 2026/04/07
 *      Author: sin
 */

#ifndef ASM09_UTIL_H_
#define ASM09_UTIL_H_

void fatal(char *str);
void print_line(void);
void f_record(void);
int delim(char c);
char * skip_white(char * ptr);
char mapdn(char c);
void error(char * str);
int emit(int byte);
void eword(int wd);
int alpha(char);
int alphan(char);
int any(char, char *);
int head(char *str1, char *str2);
void warning(char *str);
char * alloc(int nbytes);
void hexout(int byte);

int lobyte(int i);
int hibyte(int i);

#endif /* ASM09_UTIL_H_ */
