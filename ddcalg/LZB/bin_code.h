/**************************************
 *                                    *
 * Binary encoding Prototype          *
 *                                    *
 *                   Kouichi Kitamura *
 *                                    *
 **************************************/

extern int flush_need;
// extern FILE *fp_in,
extern FILE *fp_out;

extern int I_log(int num, int *filt);
extern void fit_code(int code, int filt);
extern void pos_code(int code, int p_pos);
extern void pos_code2(int code, int p_pos);
extern void leng_code(int code);
