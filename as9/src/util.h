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
int emit(int byte);
int lobyte(int i);
int hibyte(int i);
int head(char * str1, char * str2);
void eword(int wd);

int alpha(char c);
int alphan(char c);
int any(char c, char *str);

void warn(char * str);
char * alloc(int nbytes);
void hexout(int byte);
int white(char c);

#endif /* SRC_UTIL_H_ */
