/*
 *	Z80/8080-Macro-Assembler - Intel-like macro implementation
 *	Copyright (C) 2022-2024 by Thomas Eberhardt
 */

/*
 *	processing of all macro PSEUDO ops
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "z80asm.h"
#include "z80alst.h"
#include "z80anum.h"
#include "z80apfun.h"
#include "z80amfun.h"

#define CONCAT		'&'			/* parameter concatenation */
#define LITERAL		'^'			/* literal character escape */
#define BYVALUE		'%'			/* pass by value */
#define LBRACK		'<'			/* left angle bracket */
#define RBRACK		'>'			/* right angle bracket */

#define NO_LITERAL	0			/* mac_subst lit_flag values */
#define LIT_BEFORE	1

#define NO_CONCAT	0			/* mac_subst cat_flag values */
#define CAT_BEFORE	1
#define CAT_AFTER	2

typedef struct dum {				/* macro dummy */
	char *dum_name;				/* dummy name */
	struct dum *dum_next;			/* next dummy in list */
} dum_t;

typedef struct line {				/* macro source line */
	char *line_text;			/* source line */
	struct line *line_next;			/* next line in list */
} line_t;

struct expn;					/* forward declaration */

typedef void (start_func_t)(struct expn *e, char *operand);
typedef int (rept_func_t)(struct expn *e);

typedef struct mac {				/* macro */
	start_func_t *mac_start;		/* start expansion function */
	rept_func_t *mac_rept;			/* repeat expansion function */
	char *mac_name;				/* macro name */
	int mac_refflg;				/* macro reference flag */
	WORD mac_nrept;				/* REPT count */
	char *mac_irp;				/* IRP, IRPC character list */
	dum_t *mac_dums, *mac_dums_last;	/* macro dummies */
	line_t *mac_lines, *mac_lines_last;	/* macro body */
	struct mac *mac_prev, *mac_next;	/* prev./next macro in list */
} mac_t;

typedef struct parm {				/* expansion parameter */
	char *parm_name;			/* dummy name */
	char *parm_val;				/* parameter value */
	struct parm *parm_next;			/* next parameter in list */
} parm_t;

typedef struct loc {				/* expansion local label */
	char *loc_name;				/* local label name */
	char loc_val[8];			/* local label value ??xxxx */
	struct loc *loc_next;			/* next local label in list */
} loc_t;

typedef struct expn {				/* macro expansion */
	mac_t *expn_mac;			/* macro being expanded */
	parm_t *expn_parms, *expn_parms_last;	/* macro parameters */
	loc_t *expn_locs, *expn_locs_last;	/* local labels */
	line_t *expn_line;			/* current expansion line */
	int expn_cond_state[COND_STATE_SIZE];	/* cond state before expn */
	WORD expn_iter;				/* curr. expansion iteration */
	char *expn_irp;				/* IRP, IRPC character list */
	struct expn *expn_next;			/* next expansion in list */
} expn_t;

static mac_t *mac_table;		/* MACRO table */
static mac_t *mac_curr;			/* current macro */
static mac_t **mac_array;		/* sorted table for iterator */
static int mac_def_nest;		/* macro def nesting level */
static int mac_exp_nest;		/* macro exp nesting level */
static int mac_symmax;			/* max. macro len observed */
static int mac_index;			/* index for iterator */
static int mac_sort;			/* sort mode for iterator */
static int mac_count;			/* number of macros defined */
static expn_t *mac_expn;		/* macro expansion stack */
static WORD mac_loc_cnt;		/* counter for LOCAL labels */
static char tmp[MAXLINE + 1];		/* temporary buffer */

/*
 *	return current macro definition nesting level
 */
int mac_get_def_nest(void)
{
	return mac_def_nest;
}

/*
 *	return current macro expansion nesting level
 */
int mac_get_exp_nest(void)
{
	return mac_exp_nest;
}

/*
 *	return maximum macro symbol length observed
 */
int mac_get_symmax(void)
{
	return mac_symmax;
}

/*
 *	verify that s is a legal symbol, also truncates to symlen
 *	returns TRUE if legal, otherwise FALSE
 */
static int is_symbol(char *s)
{
	register int i, n;

	if (!IS_FSYM(*s))
		return FALSE;
	s++;
	i = 1;
	n = get_symlen();
	while (IS_SYM(*s)) {
		if (i++ == n)
			*s = '\0';
		s++;
	}
	return *s == '\0';
}

/*
 *	compare function for qsort of mac_array
 */
static int mac_compare(const void *p1, const void *p2)
{
	return strcmp((*(const mac_t **) p1)->mac_name,
		      (*(const mac_t **) p2)->mac_name);
}

/*
 *	get first macro name and refflg in *rp for listing
 *	sorted as specified in sort_mode
 */
char *mac_first(int sort_mode, int *rp)
{
	register mac_t *m;
	register int i;

	if (mac_count == 0)
		return NULL;
	mac_sort = sort_mode;
	switch (sort_mode) {
	case SYM_UNSORT:
		for (m = mac_table; m->mac_next != NULL; m = m->mac_next)
			;
		mac_curr = m;
		*rp = mac_curr->mac_refflg;
		return mac_curr->mac_name;
	case SYM_SORTN:
	case SYM_SORTA:
		mac_array = (mac_t **) malloc(sizeof(mac_t *) * mac_count);
		if (mac_array == NULL)
			fatal(F_OUTMEM, "sorting macro table");
		i = 0;
		for (m = mac_table; m != NULL; m = m->mac_next)
			mac_array[i++] = m;
		qsort(mac_array, mac_count, sizeof(mac_t *), mac_compare);
		mac_index = 0;
		*rp = mac_array[mac_index]->mac_refflg;
		return mac_array[mac_index]->mac_name;
	default:
		fatal(F_INTERN, "unknown sort mode in mac_first");
		break;
	}
}

/*
 *	get next macro name and refflg in *rp for listing
 */
char *mac_next(int *rp)
{
	if (mac_sort == SYM_UNSORT) {
		mac_curr = mac_curr->mac_prev;
		if (mac_curr != NULL) {
			*rp = mac_curr->mac_refflg;
			return mac_curr->mac_name;
		}
	} else if (++mac_index < mac_count) {
		*rp = mac_array[mac_index]->mac_refflg;
		return mac_array[mac_index]->mac_name;
	}
	return NULL;
}

/*
 *	allocate a new macro with optional name and
 *	start/repeat expansion function
 */
static mac_t *mac_new(const char *name, start_func_t *start, rept_func_t *rept)
{
	register mac_t *m;
	register int n;

	if ((m = (mac_t *) malloc(sizeof(mac_t))) == NULL)
		fatal(F_OUTMEM, "macro");
	if (name != NULL) {
		n = strlen(name);
		if ((m->mac_name = (char *) malloc(n + 1)) == NULL)
			fatal(F_OUTMEM, "macro name");
		strcpy(m->mac_name, name);
		if (n > mac_symmax)
			mac_symmax = n;
	} else
		m->mac_name = NULL;
	m->mac_start = start;
	m->mac_rept = rept;
	m->mac_refflg = FALSE;
	m->mac_nrept = 0;
	m->mac_irp = NULL;
	m->mac_dums_last = m->mac_dums = NULL;
	m->mac_lines_last = m->mac_lines = NULL;
	m->mac_next = m->mac_prev = NULL;
	return m;
}

/*
 *	delete a macro
 */
static void mac_delete(mac_t *m)
{
	register dum_t *d;
	register line_t *l;
	dum_t *d1;
	line_t *l1;

	for (d = m->mac_dums; d != NULL; d = d1) {
		d1 = d->dum_next;
		free(d->dum_name);
		free(d);
	}
	for (l = m->mac_lines; l != NULL; l = l1) {
		l1 = l->line_next;
		free(l->line_text);
		free(l);
	}
	if (m->mac_irp != NULL)
		free(m->mac_irp);
	if (m->mac_name != NULL)
		free(m->mac_name);
	free(m);
}

/*
 *	initialize variables at start of pass
 */
void mac_start_pass(int pass)
{
	UNUSED(pass);

	mac_loc_cnt = 0;
}

/*
 *	clean up at end of pass
 */
void mac_end_pass(int pass)
{
	register mac_t *m;

	if (pass == 1)
		while (mac_table != NULL) {
			m = mac_table->mac_next;
			mac_delete(mac_table);
			mac_table = m;
			mac_count--;
		}
}

/*
 * 	add a dummy to a macro
 */
static void mac_add_dum(mac_t *m, char *name)
{
	register dum_t *d;

	if ((d = (dum_t *) malloc(sizeof(dum_t))) == NULL)
		fatal(F_OUTMEM, "macro dummy");
	d->dum_name = strsave(name);
	d->dum_next = NULL;
	if (m->mac_dums == NULL)
		m->mac_dums = d;
	else
		m->mac_dums_last->dum_next = d;
	m->mac_dums_last = d;
}

/*
 * 	add a local to a macro expansion
 */
static void expn_add_loc(expn_t *e, char *name)
{
	register loc_t *l;
	register char *s;
	register char c;

	if ((l = (loc_t *) malloc(sizeof(loc_t))) == NULL)
		fatal(F_OUTMEM, "macro local label");
	l->loc_name = strsave(name);
	s = l->loc_val;
	*s++ = '?';
	*s++ = '?';
	if (mac_loc_cnt == 65535U) {
		*s++ = 'x';
		*s++ = 'x';
		*s++ = 'x';
		*s++ = 'x';
		asmerr(E_OUTLCL);
	} else {
		c = mac_loc_cnt >> 12;
		*s++ = c + (c < 10 ? '0' : 'a' - 10);
		c = (mac_loc_cnt >> 8) & 0xf;
		*s++ = c + (c < 10 ? '0' : 'a' - 10);
		c = (mac_loc_cnt >> 4) & 0xf;
		*s++ = c + (c < 10 ? '0' : 'a' - 10);
		c = mac_loc_cnt++ & 0xf;
		*s++ = c + (c < 10 ? '0' : 'a' - 10);
	}
	*s = '\0';
	l->loc_next = NULL;
	if (e->expn_locs == NULL)
		e->expn_locs = l;
	else
		e->expn_locs_last->loc_next = l;
	e->expn_locs_last = l;
	return;
}

/*
 *	start macro expansion
 *	assign values to parameters, save cond state
 */
static void mac_start_expn(mac_t *m, char *operand)
{
	register expn_t *e;
	register parm_t *p;
	register dum_t *d;
	expn_t *e1;

	if (mac_exp_nest == MACNEST) {
		/* abort macro expansion */
		for (e = mac_expn; e != NULL; e = e1) {
			if ((e1 = e->expn_next) == NULL)
				restore_cond_state(e->expn_cond_state);
			free(e);
		}
		/* delete unnamed macros (IRP, IRPC, REPT) */
		if (m->mac_name == NULL)
			mac_delete(m);
		asmerr(E_MACNST);
		return;
	}
	if ((e = (expn_t *) malloc(sizeof(expn_t))) == NULL)
		fatal(F_OUTMEM, "macro expansion");
	e->expn_mac = m;
	e->expn_parms_last = e->expn_parms = NULL;
	for (d = m->mac_dums; d != NULL; d = d->dum_next) {
		p = (parm_t *) malloc(sizeof(parm_t));
		if (p == NULL)
			fatal(F_OUTMEM, "macro parameter");
		p->parm_name = d->dum_name;
		p->parm_val = NULL;
		p->parm_next = NULL;
		if (e->expn_parms == NULL)
			e->expn_parms = p;
		else
			e->expn_parms_last->parm_next = p;
		e->expn_parms_last = p;
	}
	e->expn_locs_last = e->expn_locs = NULL;
	e->expn_line = m->mac_lines;
	save_cond_state(e->expn_cond_state);
	e->expn_iter = 0;
	e->expn_irp = m->mac_irp;
	m->mac_refflg = TRUE;
	(*m->mac_start)(e, operand);
	e->expn_next = mac_expn;
	mac_expn = e;
	mac_exp_nest++;
}

/*
 *	end macro expansion
 *	delete parameters and local labels, restore cond state
 */
static void mac_end_expn(void)
{
	register parm_t *p;
	register loc_t *l;
	register expn_t *e;
	mac_t *m;
	parm_t *p1;
	loc_t *l1;

	e = mac_expn;
	for (p = e->expn_parms; p != NULL; p = p1) {
		p1 = p->parm_next;
		if (p->parm_val != NULL)
			free(p->parm_val);
		free(p);
	}
	for (l = e->expn_locs; l != NULL; l = l1) {
		l1 = l->loc_next;
		free(l->loc_name);
		free(l);
	}
	restore_cond_state(e->expn_cond_state);
	m = e->expn_mac;
	mac_expn = e->expn_next;
	mac_exp_nest--;
	free(e);
	/* delete unnamed macros (IRP, IRPC, REPT) */
	if (m->mac_name == NULL)
		mac_delete(m);
}

/*
 *	repeat macro for IRP, IRPC, REPT when end reached
 *	end expansion for MACRO
 */
static int mac_rept_expn(void)
{
	register expn_t *e;
	register loc_t *l;
	register mac_t *m;
	loc_t *l1;

	e = mac_expn;
	e->expn_iter++;
	m = e->expn_mac;
	if (m->mac_rept != NULL && (*m->mac_rept)(e)) {
		for (l = e->expn_locs; l != NULL; l = l1) {
			l1 = l->loc_next;
			free(l->loc_name);
			free(l);
		}
		e->expn_locs_last = e->expn_locs = NULL;
		e->expn_line = m->mac_lines;
		restore_cond_state(e->expn_cond_state);
		return TRUE;
	} else {
		mac_end_expn();
		return FALSE;
	}
}

/*
 *	add source line l to current macro definition
 */
void mac_add_line(opc_t *op, char *line)
{
	register line_t *l;
	register mac_t *m;

	if ((l = (line_t *) malloc(sizeof(line_t))) == NULL)
		fatal(F_OUTMEM, "macro body line");
	l->line_text = strsave(line);
	l->line_next = NULL;
	m = mac_curr;
	if (m->mac_lines == NULL)
		m->mac_lines = l;
	else
		m->mac_lines_last->line_next = l;
	m->mac_lines_last = l;
	if (op != NULL) {
		if (op->op_flags & OP_MDEF)
			mac_def_nest++;
		else if (op->op_flags & OP_MEND) {
			if (--mac_def_nest == 0) {
				m = mac_curr;
				mac_curr = NULL;
				/* start expansion for IRP, IRPC, REPT */
				if (m->mac_name == NULL)
					mac_start_expn(m, NULL);
			}
		}
	}
}

/*
 *	get value of dummy s, NULL if not found
 */
static const char *mac_get_dummy(expn_t *e, char *s)
{
	register parm_t *p;

	for (p = e->expn_parms; p != NULL; p = p->parm_next)
		if (strcmp(p->parm_name, s) == 0)
			return p->parm_val == NULL ? "" : p->parm_val;
	return NULL;
}

/*
 *	get value of local label s, NULL if not found
 */
static const char *mac_get_local(expn_t *e, char *s)
{
	register loc_t *l;

	for (l = e->expn_locs; l != NULL; l = l->loc_next)
		if (strcmp(l->loc_name, s) == 0)
			return l->loc_val;
	return NULL;
}

/*
 *	substitute dummies or locals with actual values in source line s
 *	returns the result in t
 */
static void mac_subst(char *t, char *s, expn_t *e,
		      const char *(*getf)(expn_t *e, char *s))
{
	register const char *v;
	register int m;
	register int cat_flag;
	char *s1, *t0, *t1, c;
	int lit_flag;

	if (*s == LINCOM || (*s == LINOPT && !IS_SYM(*(s + 1)))) {
		strcpy(t, s);
		return;
	}
	t0 = t;
	cat_flag = NO_CONCAT;
	lit_flag = NO_LITERAL;
	while (*s != '\0') {
		if (IS_FSYM(*s)) {
			/* gather symbol */
			s1 = s;
			t1 = t;
			*t++ = TO_UPP(*s);
			s++;
			while (IS_SYM(*s)) {
				*t++ = TO_UPP(*s);
				s++;
			}
			*t = '\0';
			v = (*getf)(e, t1);
			/* don't substitute dummy if leading LITERAL */
			if (v == NULL || lit_flag == LIT_BEFORE) {
				t = s;
				s = s1;
				s1 = t;
				t = t1;
				/* remove leading LITERAL if dummy */
				if (v != NULL && lit_flag == LIT_BEFORE)
					t--;
				while (s < s1)
					*t++ = *s++;
				cat_flag = NO_CONCAT;
				lit_flag = NO_LITERAL;
				continue;
			}
			/* substitute dummy */
			t = t1;
			/* remove leading CONCAT */
			if (cat_flag == CAT_BEFORE)
				t--;
			m = MAXLINE - (t - t0);
			while (*v != '\0') {
				if (m-- == 0) {
					asmerr(E_MACOVF);
					*t = '\0';
					return;
				}
				*t++ = *v++;
			}
			/* skip trailing CONCAT */
			if (*s == CONCAT) {
				cat_flag = CAT_AFTER;
				s++;
			} else
				cat_flag = NO_CONCAT;
			lit_flag = NO_LITERAL;
		} else if (*s == STRDEL || *s == STRDEL2) {
			*t++ = c = *s++;
			cat_flag = NO_CONCAT;
			while (TRUE) {
				if (*s == '\0') {
					/* undelimited, don't complain here,
					   could be EX AF,AF' */
					break;
				} else if (*s == c) {
					cat_flag = NO_CONCAT;
					*t++ = *s++;
					if (*s != c) /* double delim? */
						break;
					else {
						*t++ = *s++;
						continue;
					}
				} else if (!IS_FSYM(*s)) {
					if (*s == CONCAT)
						cat_flag = CAT_BEFORE;
					else
						cat_flag = NO_CONCAT;
					*t++ = *s++;
					continue;
				}
				/* gather symbol */
				t1 = t;
				*t++ = *s++;
				while (IS_SYM(*s))
					*t++ = *s++;
				/* subst. poss. dummy if CONCAT before/after */
				if (cat_flag != NO_CONCAT || *s == CONCAT) {
					*t = '\0';
					v = (*getf)(e, t1);
					/* not a dummy? */
					if (v == NULL) {
						cat_flag = NO_CONCAT;
						continue;
					}
					/* substitute dummy */
					t = t1;
					/* remove leading CONCAT */
					if (cat_flag == CAT_BEFORE)
						t--;
					m = MAXLINE - (t - t0);
					while (*v != '\0') {
						if (m-- == 0) {
							asmerr(E_MACOVF);
							*t = '\0';
							return;
						}
						*t++ = *v++;
					}
					/* skip trailing CONCAT */
					if (*s == CONCAT) {
						cat_flag = CAT_AFTER;
						s++;
					} else
						cat_flag = NO_CONCAT;
				}
			}
			lit_flag = NO_LITERAL;
		} else if (*s == COMMENT) {
			/* don't copy double COMMENT comments */
			if (*(s + 1) == COMMENT)
				*t = '\0';
			else
				strcpy(t, s);
			return;
		} else {
			cat_flag = NO_CONCAT;
			lit_flag = NO_LITERAL;
			if (*s == CONCAT)
				cat_flag = CAT_BEFORE;
			else if (*s == LITERAL)
				lit_flag = LIT_BEFORE;
			*t++ = *s++;
		}
	}
	*t = '\0';
}

/*
 *	get next macro expansion line
 */
char *mac_expand(char *line)
{
	register expn_t *e;

	e = mac_expn;
	if (e->expn_line == NULL && !mac_rept_expn())
		return NULL;
	/* first substitute dummies with parameter values */
	mac_subst(tmp, e->expn_line->line_text, e, mac_get_dummy);
	/* next substitute local labels with ??xxxx */
	mac_subst(line, tmp, e, mac_get_local);
	e->expn_line = e->expn_line->line_next;
	return line;
}

/*
 *	macro lookup
 */
int mac_lookup(char *sym_name)
{
	register mac_t *m;

	mac_curr = NULL;
	for (m = mac_table; m != NULL; m = m->mac_next)
		if (strcmp(m->mac_name, sym_name) == 0) {
			mac_curr = m;
			return TRUE;
		}
	return FALSE;
}

/*
 *	MACRO invocation, to be called after successful mac_lookup()
 */
void mac_call(char *operand)
{
	if (mac_curr != NULL) {
		mac_start_expn(mac_curr, operand);
		mac_curr = NULL;
	} else
		fatal(F_INTERN, "mac_call with no macro");
}

/*
 *	get next macro parameter
 */
static char *mac_next_parm(char *s)
{
	register char *t, c;
	register int m;
	char *t1;
	int n, r;
	WORD w, v;

	t1 = t = tmp;
	n = 0;		/* angle brackets nesting level */
	while (IS_SPC(*s))
		s++;
	while (*s != '\0') {
		if (*s == STRDEL || *s == STRDEL2) {
			/* keep delimiters, but reduce double delimiters */
			*t++ = c = *s++;
			while (TRUE) {
				if (*s == '\0') {
					asmerr(E_MISDEL);
					return NULL;
				} else if (*s == c) {
					if (*(s + 1) != c) /* double delim? */
						break;
					else
						s++;
				}
				*t++ = *s++;
			}
			*t++ = *s++;
		} else if (*s == BYVALUE && n == 0) {
			/* pass by value */
			s++;
			t1 = t;
			while (*s != '\0' && *s != ',' && *s != COMMENT) {
				if (*s == STRDEL || *s == STRDEL2) {
					*t++ = c = *s++;
					while (TRUE) {
						if (*s == '\0') {
							asmerr(E_MISDEL);
							return NULL;
						} else if (*s == c) {
							*t++ = *s++;
							if (*s != c)
								break;
						}
						*t++ = *s++;
					}
				} else {
					*t++ = TO_UPP(*s);
					s++;
				}
			}
			*t = '\0';
			v = w = eval(t1);
			/* count digits in current radix */
			r = get_radix();
			for (m = 1; v > r; m++)
				v /= r;
			if (v > 9)
				m++;
			t1 += m;
			if (t1 - tmp > MAXLINE) {
				asmerr(E_MACOVF);
				return NULL;
			}
			/* generate digits backwards in current radix */
			for (t = t1; m > 0; m--) {
				v = w % r;
				*--t = v + (v < 10 ? '0' : 'a' - 10);
				w /= r;
			}
			break;
		} else if (*s == LITERAL) {
			/* literalize next character */
			s++;
			if (*s == '\0') {
				asmerr(E_INVOPE);
				return NULL;
			} else
				*t++ = *s++;
		} else if (*s == LBRACK) {
			/* remove top level LBRACK */
			if (n > 0)
				*t++ = *s++;
			else
				s++;
			n++;
		} else if (*s == RBRACK) {
			if (n == 0) {
				asmerr(E_INVOPE);
				return NULL;
			} else {
				/* remove top level RBRACK */
				n--;
				if (n > 0)
					*t++ = *s++;
				else
					s++;
			}
		} else if ((*s == ',' || *s == COMMENT) && n == 0)
			break;
		else
			*t++ = *s++;
		t1 = t;
		while (IS_SPC(*s))
			*t++ = *s++;
	}
	if (n > 0) {
		asmerr(E_INVOPE);
		return NULL;
	}
	*t1 = '\0';
	return s;
}

/*
 *	start IRP macro expansion
 */
static void mac_start_irp(expn_t *e, char *operand)
{
	register char *s;

	UNUSED(operand);

	if (*e->expn_irp != '\0') {
		if ((s = mac_next_parm(e->expn_irp)) != NULL) {
			e->expn_irp = s;
			e->expn_parms->parm_val = strsave(tmp);
		}
	}
}

/*
 *	repeat IRP macro expansion
 */
static int mac_rept_irp(expn_t *e)
{
	register char *s;

	s = e->expn_irp;
	if (*s == '\0')
		return FALSE;
	else if (*s++ != ',') {
		asmerr(E_INVOPE);
		return FALSE;
	} else {
		if ((s = mac_next_parm(s)) == NULL)
			return FALSE;
		e->expn_irp = s;
		if (e->expn_parms->parm_val != NULL)
			free(e->expn_parms->parm_val);
		e->expn_parms->parm_val = strsave(tmp);
		return TRUE;
	}
}

/*
 *	start IRPC macro expansion
 */
static void mac_start_irpc(expn_t *e, char *operand)
{
	register char *s;

	UNUSED(operand);

	if (*e->expn_irp != '\0') {
		if ((s = (char *) malloc(2)) == NULL)
			fatal(F_OUTMEM, "IRPC character");
		*s = *e->expn_irp++;
		*(s + 1) = '\0';
		e->expn_parms->parm_val = s;
	}
}

/*
 *	repeat IRPC macro expansion
 */
static int mac_rept_irpc(expn_t *e)
{
	if (*e->expn_irp != '\0') {
		*e->expn_parms->parm_val = *e->expn_irp++;
		return TRUE;
	} else
		return FALSE;
}

/*
 *	start MACRO macro expansion
 */
static void mac_start_macro(expn_t *e, char *operand)
{
	register char *s;
	register parm_t *p;

	s = operand;
	p = e->expn_parms;
	while (p != NULL && *s != '\0' && *s != COMMENT) {
		if ((s = mac_next_parm(s)) == NULL)
			return;
		if (*s == ',')
			s++;
		else if (*s != '\0' && *s != COMMENT) {
			asmerr(E_INVOPE);
			return;
		}
		p->parm_val = strsave(tmp);
		p = p->parm_next;
	}
}

/*
 *	start REPT macro expansion
 */
static void mac_start_rept(expn_t *e, char *operand)
{
	UNUSED(operand);

	if (e->expn_mac->mac_nrept == 0)
		e->expn_line = NULL;
}

/*
 *	repeat REPT macro expansion
 */
static int mac_rept_rept(expn_t *e)
{
	return e->expn_iter < e->expn_mac->mac_nrept;
}

/*
 *	IFB, IFNB
 */
int mac_op_ifb(char *operand, int *false_sect_flagp)
{
	register char *s;
	register int err = FALSE;

	if ((s = mac_next_parm(operand)) != NULL) {
		if (*s == '\0' || *s == COMMENT) {
			if (*tmp != '\0')
				*false_sect_flagp = TRUE;
		} else {
			asmerr(E_INVOPE);
			err = TRUE;
		}
	} else
		err = TRUE;
	return err;
}

/*
 *	IFIDN, IFDIF
 */
int mac_op_ifidn(char *operand, int *false_sect_flagp)
{
	register char *s, *t = NULL;
	register int err = FALSE;

	if ((s = mac_next_parm(operand)) != NULL) {
		if (*s++ == ',') {
			t = strsave(tmp);
			if ((s = mac_next_parm(s)) != NULL) {
				if ((*s == '\0' || *s == COMMENT)) {
					if (strcmp(t, s) != 0)
						*false_sect_flagp = TRUE;
				} else {
					asmerr(E_INVOPE);
					err = TRUE;
				}
			} else
				err = TRUE;
		} else {
			asmerr(E_MISOPE);
			err = TRUE;
		}
	} else
		err = TRUE;
	if (t != NULL)
		free(t);
	return err;
}

/*
 *	ENDM
 */
WORD op_endm(int pass, BYTE dummy1, BYTE dummy2, char *operand, BYTE *ops)
{
	UNUSED(pass);
	UNUSED(dummy1);
	UNUSED(dummy2);
	UNUSED(operand);
	UNUSED(ops);

	if (mac_exp_nest == 0)
		asmerr(E_NIMEXP);
	else
		(void) mac_rept_expn();
	return 0;
}

/*
 *	EXITM
 */
WORD op_exitm(int pass, BYTE dummy1, BYTE dummy2, char *operand, BYTE *ops)
{
	UNUSED(pass);
	UNUSED(dummy1);
	UNUSED(dummy2);
	UNUSED(operand);
	UNUSED(ops);

	if (mac_exp_nest == 0)
		asmerr(E_NIMEXP);
	else
		mac_end_expn();
	return 0;
}

/*
 *	IRP and IRPC
 */
WORD op_irp(int pass, BYTE op_code, BYTE dummy, char *operand, BYTE *ops)
{
	register char *s, *t;
	register int i;
	mac_t *m;
	int err = FALSE, n;

	UNUSED(pass);
	UNUSED(dummy);
	UNUSED(ops);

	switch (op_code) {
	case 1:				/* IRP */
		m = mac_new(NULL, mac_start_irp, mac_rept_irp);
		break;
	case 2:				/* IRPC */
		m = mac_new(NULL, mac_start_irpc, mac_rept_irpc);
		break;
	default:
		fatal(F_INTERN, "invalid opcode for function op_irp");
		break;
	}
	s = operand;
	t = tmp;
	if (IS_FSYM(*s)) {
		*t++ = TO_UPP(*s);
		s++;
		i = 1;
		n = get_symlen();
		while (IS_SYM(*s)) {
			if (i++ < n)
				*t++ = TO_UPP(*s);
			s++;
		}
		*t = '\0';
		mac_add_dum(m, tmp);
		while (IS_SPC(*s))
			s++;
		if (*s++ == ',') {
			while (IS_SPC(*s))
				s++;
			if ((s = mac_next_parm(s)) != NULL) {
				if (*s == '\0' || *s == COMMENT) {
					m->mac_irp = strsave(tmp);
					mac_curr = m;
					mac_def_nest++;
				} else {
					asmerr(E_INVOPE);
					err = TRUE;
				}
			} else
				err = TRUE;
		} else {
			asmerr(E_MISOPE);
			err = TRUE;
		}
	} else {
		asmerr(E_INVOPE);
		err = TRUE;
	}
	if (err)
		mac_delete(m);
	return 0;
}

/*
 *	LOCAL
 */
WORD op_local(int pass, BYTE dummy1, BYTE dummy2, char *operand, BYTE *ops)
{
	register char *s, *s1;

	UNUSED(pass);
	UNUSED(dummy1);
	UNUSED(dummy2);
	UNUSED(ops);

	if (mac_exp_nest == 0) {
		asmerr(E_NIMEXP);
		return 0;
	}
	s = operand;
	while (s != NULL) {
		s1 = next_arg(s, NULL);
		if (*s != '\0') {
			if (is_symbol(s))
				expn_add_loc(mac_expn, s);
			else {
				asmerr(E_INVOPE);
				break;
			}
		}
		s = s1;
	}
	return 0;
}

/*
 *	MACRO
 */
WORD op_macro(int pass, BYTE dummy1, BYTE dummy2, char *operand, BYTE *ops)
{
	register char *s, *s1;
	register mac_t *m;

	UNUSED(pass);
	UNUSED(dummy1);
	UNUSED(dummy2);
	UNUSED(ops);

	m = mac_new(get_label(), mac_start_macro, NULL);
	if (mac_table != NULL)
		mac_table->mac_prev = m;
	m->mac_next = mac_table;
	mac_table = m;
	mac_count++;
	s = operand;
	while (s != NULL) {
		s1 = next_arg(s, NULL);
		if (*s != '\0') {
			if (is_symbol(s))
				mac_add_dum(m, s);
			else {
				asmerr(E_INVOPE);
				break;
			}
		}
		s = s1;
	}
	mac_curr = m;
	mac_def_nest++;
	return 0;
}

/*
 *	REPT
 */
WORD op_rept(int pass, BYTE dummy1, BYTE dummy2, char *operand, BYTE *ops)
{
	register mac_t *m;

	UNUSED(pass);
	UNUSED(dummy1);
	UNUSED(dummy2);
	UNUSED(ops);

	m = mac_new(NULL, mac_start_rept, mac_rept_rept);
	m->mac_nrept = eval(operand);
	mac_curr = m;
	mac_def_nest++;
	return 0;
}
