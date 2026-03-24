/*
 * pseudo.h
 *
 *  Created on: 2025/11/15
 *      Author: sin
 */

#ifndef SRC_PSEUDO_H_
#define SRC_PSEUDO_H_

#define RMB     0       /* Reserve Memory Bytes         */
#define FCB     1       /* Form Constant Bytes          */
#define FDB     2       /* Form Double Bytes (words)    */
#define FCC     3       /* Form Constant Characters     */
#define ORG     4       /* Origin                       */
#define EQU     5       /* Equate                       */
#define ZMB     6       /* Zero memory bytes            */
#define FILL    7       /* block fill constant bytes    */
#define OPT     8       /* assembler option             */
#define NULL_OP 9       /* null pseudo op               */
#define PAGE    10      /* new page                     */
#define SET     11      /* Set ("re-Equate")            */
#define FCCZ    12      /* Form Constant Characters     */

struct oper pseudo[] = {
		{"bsz",  PSEUDO, ZMB,    0},
		{"end",  PSEUDO, NULL_OP,0},
		{"equ",  PSEUDO, EQU,    0},
		{"fcb",  PSEUDO, FCB,    0},
		{"fcc",  PSEUDO, FCC,    0},
		{"fccz",  PSEUDO, FCCZ,  0},
		{"fdb",  PSEUDO, FDB,    0},
		{"fill", PSEUDO, FILL,   0},
		{"nam",  PSEUDO, NULL_OP,0},
		{"name", PSEUDO, NULL_OP,0},
		{"opt",  PSEUDO, OPT,    0},
		{"org",  PSEUDO, ORG,    0},
		{"pag",  PSEUDO, PAGE,   0},
		{"page", PSEUDO, PAGE,   0},
		{"rmb",  PSEUDO, RMB,    0},
		{"set",  PSEUDO, SET,    0},
		{"spc",  PSEUDO, NULL_OP,0},
		{"ttl",  PSEUDO, NULL_OP,0},
		{"zmb",  PSEUDO, ZMB,    0},
};

void do_pseudo(int op);


#endif /* SRC_PSEUDO_H_ */
