/*
 * symtab.h
 *
 *  Created on: 2026/04/07
 *      Author: sin
 */

#ifndef ASM09_SYMTAB_H_
#define ASM09_SYMTAB_H_


int install(char *str, int val, int override);
struct oper * mne_look(char * str);
struct nlist * lookup(char * name);

#endif /* ASM09_SYMTAB_H_ */
