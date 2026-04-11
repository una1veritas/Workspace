/*
 * symtab.h
 *
 *  Created on: 2026/04/07
 *      Author: sin
 */

#ifndef ASM09_SYMTAB_H_
#define ASM09_SYMTAB_H_


int install(const std::set<namedef> & namelist, const char *str, const int val, const int override);
struct oper * mne_look(char * str);
const namedef * lookup(const std::set<namedef> & namelist, const char * name);

#endif /* ASM09_SYMTAB_H_ */
