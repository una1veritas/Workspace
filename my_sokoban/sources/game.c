/*
** game.c for  in /home/kawrantin/delivery/PSU_2016_my_sokoban
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Fri Dec 16 13:39:30 2016 Le Goff Kawrantin
** Last update Tue Dec 20 22:43:05 2016 Le Goff Kawrantin
*/

#include <ncurses/ncurses.h>
#include "include/my.h"

int	init_player(char **map, int lines, int cols, struct s_map *m)
{
  int	j;
  int	i;

  j = 0;
  i = 0;
  while (j < lines)
    {
      if (map[j][i] == 'P')
	{
	  map[j][i] = ' ';
	  m->init_map[j][i] = ' ';
	}
      i++;
      if (i == cols)
	{
	  i = 0;
	  j++;
	}
    }
  return (0);
}

int	ncurses_trash(struct s_map *m, char **map, int lines, int cols)
{
  initscr();
  curs_set(FALSE);
  check_O(m, map);
  check_X(m, map);
  is_victory(m, map);
  errors(lines, cols);
  return (0);
}

int     game(char **map, struct s_map *m)
{
  int   lines;
  int   cols;
  int   i;

  lines = m->lines;
  cols = m->cols;
  i = -1;
  get_pos(map, lines, cols, m);
  m->init_X = m->X;
  m->init_Y = m->Y;
  init_player(map, lines, cols, m);
  while (1)
    {
      ncurses_trash(m, map, lines, cols);
      while (++i < lines)
	mvprintw(i, 0, map[i]);
      i = -1;
      mvprintw(m->Y, m->X, "P");
      map = key_event(m, map, lines);
      clear();
      refresh();
    }
  endwin();
  return (0);
}
