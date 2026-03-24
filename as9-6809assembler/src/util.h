/*
 * util.h
 *
 *  Created on: 2025/11/13
 *      Author: sin
 */

#ifndef SRC_UTIL_H_
#define SRC_UTIL_H_


void fatal(char	*str);
void print_line();
void f_record();
int delim(char c);
char* skip_white(char * ptr);
char mapdn(char c);
void error(char * str);
unsigned char emit(int byte);
unsigned char lobyte(int i);
unsigned char hibyte(int i);
int head(char * str1, char * str2);
void eword(int wd);

unsigned char alpha(char c);
unsigned char alphan(char c);
unsigned char any(char c, char *str);

void warn(char * str);
char * alloc(int nbytes);

#endif /* SRC_UTIL_H_ */
