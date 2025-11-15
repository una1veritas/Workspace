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
void emit(int byte);
unsigned char lobyte(int i);
unsigned char hibyte(int i);
int head(char * str1, char * str2);
void eword(int wd);


#endif /* SRC_UTIL_H_ */
