/*
 * do9.h
 *
 *  Created on: 2025/11/15
 *      Author: sin
 */

#ifndef SRC_DO9_H_
#define SRC_DO9_H_


void localinit();
void do_op(int opcode, int class);
void do_gen(int op, int mode);
int set_mode();

unsigned char regnum(void);
void do_indexed(int op);
unsigned char rtype(int r);
void abd_index(int pbyte);

#endif /* SRC_DO9_H_ */
