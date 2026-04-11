/*
 * do9.h
 *
 *  Created on: 2026/04/07
 *      Author: sin
 */

#ifndef ASM09_DO9_H_
#define ASM09_DO9_H_

void localinit(void);
void do_op(int opcode, int opclass);
void do_gen(int op, int mode);
void do_indexed(int);
void abd_index(int );
int rtype(int );

int set_mode(void);
int regnum(void);

#endif /* ASM09_DO9_H_ */
