/**********************************
 *                                *
 * LZB Encode Prototype           *
 *                                *
 *	         Kouichi Kitamura     *
 *                                *
 **********************************/

extern int get_c(void);
extern void put_code(int h, int pos, int lng, int part);
extern int get_buf(int pos);
extern int put_buf(int c);
extern void fill_buf(int c);
extern int get_real_pos(int pos);
extern int get_ring_pos(int pos);
extern int get_buf_leng(void);
extern int move_part(int num);
extern int get_search_code(int pos);
extern int get_part(void);
extern void pre_set_buf(void);
extern int set_buf(void);
extern void init_buf(void);
extern void mod_buf(int num);
extern int do_cont(void);
extern int search(int pos0, int pos1, int *lng);
extern int set_tree(int pos, int *match_pos, int *match_leng);
extern void encode(int ps, int pe);
extern void del_tree(int shift);
//extern void main(int argc, char **argv);
