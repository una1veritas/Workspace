/*
** my.h for my in /home/kawrantin/delivery/CPool_Day09/include
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Thu Oct 13 09:52:01 2016 Le Goff Kawrantin
** Last update Thu Jan 12 15:46:35 2017 Le Goff Kawrantin
*/

#ifndef MY_H_
# define MY_H_

#include <stdarg.h>
#include <ncurses/ncurses.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>

struct s_map
{
  int	lines;
  int	cols;
  int	X;
  int	Y;
  int	init_X;
  int	init_Y;
  int	**pos_O;
  int	size_tab;
  char	**init_map;
};

# define WIDE COLS / 2 - my_strlen(map[1]) / 2
# define HIGH LINES / 2 - lines / 2

int	print_help();
char	**open_map(char *path, struct s_map *m);
int	nbr_lines(char *buffer);
int	max_length(char *buffer);
int	map_lines(char **map);
int	map_cols(char **map);
int	get_pos(char **map, int lines, int cols, struct s_map *m);
int	game(char **map, struct s_map *m);
int	move_up(struct s_map *m, char **map);
int	move_down(struct s_map *m, char **map);
int	move_right(struct s_map *m, char **map);
int	move_left(struct s_map *m, char **map);
int	file_size(char *path);
int	detect_gameover_trash(char **map, int i, int j, int a);
int	detect_gameover(char **map, int i, int j, struct s_map *m);
int	check_O(struct s_map *m, char **map);
int	check_X(struct s_map *m, char **map);
char	**restart(char **map, int lines, struct s_map *m);
char	**key_event(struct s_map *m, char **map, int lines);
int	errors(int lines, int cols);
int	is_victory(struct s_map *m, char **map);

void	my_putchar(char c);
int	my_isneg(int nb);
int	my_put_nbr(int nb);
int	my_swap(int nb);
int	my_putstr(char *str);
int	my_strlen(char *str);
int	my_getnbr(char *str);
void	my_sort_int_tab(int *tab, int size);
int	my_power_rec(int nb, int power);
int	my_square_root(int tab);
int	my_is_prime(int nbr);
int	my_find_prime(int nb);
char	*my_strcpy(char *dest, char *src);
char	*my_strncpy(char *dest, char *src, int n);
char	*my_revstr(char *str);
char	*mu_strstr(char *str, char *to_find);
int	my_strcmp(char *s1, char *s2);
int	my_strncmp(char *s1, char *s2, int n);
char	*my_strupcase(char *str);
char	*my_strlowcase(char *str);
char	*my_strcapitalize(char *str);
int	my_str_isalpha(char *str);
int	my_str_isnum(char *str);
int	my_str_islower(char *str);
int	my_str_isupper(char *str);
int	my_str_isprintable(char *str);
int	my_showstr(char *str);
int	my_showmem(char *str, int size);
char	*my_strcat(char *dest, char *src);
char	*my_strncat(char *dest, char *src, int nb);

int     my_printf(int final_len, int i, char *format, ...);
int     my_put_nbr_va(va_list *ap);
int     my_putstr_va(va_list *ap);
int     my_putstr_non_print(va_list *ap);
int     my_put_nbr_unsigned(va_list *ap);
int     my_putchar_va(va_list *ap);
int     my_put_nbr_bin(va_list *ap);
int     my_put_nbr_oct(va_list *ap);
int     my_put_nbr_hex_up(va_list *ap);
int     my_put_nbr_hex_low(va_list *ap);
int     the_little_mod_function_with_a_big_name(int final_len);
int     my_put_address(va_list *ap);

#endif /* !MY_H_ */
