/**********************************
 *                                *
 * Binary Search Tree Operation   *
 *                                *
 *               Kouichi Kitamura *
 *                                *
 **********************************/

#include <stdio.h>
#include <stdlib.h>

#include "tree.h"

#include "swatch.h"

Tree *talloc(void) {
	return ((Tree *) malloc(sizeof(Tree)));
}

//void insert(x1, x2, p, b_ps, b_pe, ret, leng)	/* 4記号の挿入 */
void insert(unsigned long x1, /* retはこれまでの中で */
unsigned char x2, /* もっとも最近のその記号の絶対バッファ位置 */
Tree **p, /* 新規なら -1 */
int *b_ps, int b_pe, int *ret, int *leng /* lengは一致記号数(2~4) */
) {
	if ((*p) == NULL) {
		sw_reset(ALLOC);
		(*p) = talloc();
		sw_accumlate(ALLOC);
		if ((*p) == NULL) {
			fprintf(stderr, "MEMORY FULL!\n");
			exit(1);
		}
		(*p)->key1 = x1;
		(*p)->key2 = x2;
		(*p)->left = (*p)->right = NULL;
		(*p)->buf_start_pos = b_pe;
		(*p)->buf_end_pos = b_pe;
		(*ret) = -1;
	} else {
		if (x1 == (*p)->key1) {
			(*leng) = 3;
			(*b_ps) = (*p)->buf_start_pos; /* 3記号一致 */
			if (x2 == (*p)->key2) { /* 4記号一致 */
				(*leng) = 4;
				(*ret) = (*p)->buf_end_pos;
				(*p)->buf_end_pos = b_pe;
			} else if (x2 < (*p)->key2)
				insert(x1, x2, &((*p)->left), b_ps, b_pe, ret, leng);
			else
				insert(x1, x2, &((*p)->right), b_ps, b_pe, ret, leng);
		} else {
			if ((x1 & 0x0ff0) == ((*p)->key1 & 0x0ff0)) { /* 2記号の一致 */
				(*leng) = 2;
				(*b_ps) = (*p)->buf_start_pos;
			}
			if (x1 < (*p)->key1)
				insert(x1, x2, &((*p)->left), b_ps, b_pe, ret, leng);
			else
				insert(x1, x2, &((*p)->right), b_ps, b_pe, ret, leng);
		}
	}
}

void delete( //x1, x2, p, b_p)	/* 4記号の削除 */
		unsigned long x1, unsigned char x2, Tree **p, int b_p) {
	Tree *p1, *p2, *p3;
	if ((*p) == NULL) {
		fprintf(stderr, "DELETE SEARCH FAULT.\n");
		exit(1);
	} else if ((*p)->key1 == x1) {
		if ((*p)->key2 == x2) {
			if (b_p == -1) {
				p3 = (*p);
				if ((*p)->right == NULL)
					(*p) = (*p)->left;
				else if ((*p)->left == NULL)
					(*p) = (*p)->right;
				else {
					p2 = (*p)->right;
					if (p2->left == NULL) {
						p2->left = (*p)->left;
						(*p) = p2;
					} else {
						while (p2->left != NULL) {
							p1 = p2;
							p2 = p2->left;
						}
						p1->left = p2->right;
						p2->left = (*p)->left;
						p2->right = (*p)->right;
						(*p) = p2;
					}
				}
				sw_reset(ALLOC);
				free(p3);
				sw_accumlate(ALLOC);
			} else
				(*p)->buf_start_pos = b_p;
		} else if (x2 < (*p)->key2)
			delete(x1, x2, &((*p)->left), b_p);
		else
			delete(x1, x2, &((*p)->right), b_p);
	} else if (x1 < (*p)->key1)
		delete(x1, x2, &((*p)->left), b_p);
	else
		delete(x1, x2, &((*p)->right), b_p);
}
