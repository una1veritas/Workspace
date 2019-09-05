/*
** victory_error.c for  in /home/kawrantin/delivery/PSU_2016_my_sokoban
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Tue Dec 20 21:57:37 2016 Le Goff Kawrantin
** Last update Tue Dec 20 21:57:38 2016 Le Goff Kawrantin
*/

#include "include/my.h"

int     errors(int lines, int cols)
{
  while (LINES < lines || COLS < cols)
    {
      clear();
      mvprintw(LINES / 2, COLS / 2 - 6, "RESIZE WINDOW");
      if (getch() == 'q')
	{
	  endwin();
	  exit(0);
	}
      refresh();
    }
  clear();
  return (0);
}

int     is_victory(struct s_map *m, char **map)
{
  int   i;
  int   j;
  int   nbr_O;

  i = j = nbr_O = 0;
  while (j < m->lines)
    {
      if (map[j][i++] == 'O')
	nbr_O++;
      if (i == m->cols)
	{
	  i = 0;
	  j++;
	}
    }
  if (nbr_O == 0)
    {
      endwin();
      exit(0);
    }
  return (0);
}
