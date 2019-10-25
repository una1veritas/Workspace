/*
** key_events.c for  in /home/kawrantin/delivery/PSU_2016_my_sokoban
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Tue Dec 20 21:47:41 2016 Le Goff Kawrantin
** Last update Tue Dec 20 22:04:48 2016 Le Goff Kawrantin
*/

#include "include/my.h"

char    **restart(char **map, int lines, struct s_map *m)
{
  int   i;
  int   j;

  i = j = 0;
  m->X = m->init_X;
  m->Y = m->init_Y;
  while (j < lines)
    {
      map[j][i] = m->init_map[j][i];
      i++;
      if (i == m->cols)
	{
	  i = 0;
	  j++;
	}
    }
  return (map);
}

char    **key_event(struct s_map *m, char **map, int lines)
{
  int   ch;

  ch = getch();
  if (ch == 'A')
    move_up(m, map);
  if (ch == 'B')
    move_down(m, map);
  if (ch == 'C')
    move_right(m, map);
  if (ch == 'D')
    move_left(m, map);
  if (ch == ' ')
    map = restart(map, lines, m);
  if (m->X >= m->cols)
    m->X = m->X - 1;
  if (m->Y >= m->lines)
    m->Y = m->Y - 1;
  if (ch == 'q')
    {
      endwin();
      exit(0);
    }
  return (map);
}
