/*
 * symtab.h
 *
 *  Created on: 2026/04/07
 *      Author: sin
 */

#ifndef ASM09_SYMTAB_H_
#define ASM09_SYMTAB_H_


int install(const char *str, const int val, const bool override);
struct oper * mne_look(const char * str);
struct nlist * lookup(const char * name);


void free_symtab(struct nlist *ptr);

#endif /* ASM09_SYMTAB_H_ */
