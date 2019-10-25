/*
** get_pos.c for  in /home/kawrantin/delivery/PSU_2016_my_sokoban
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Fri Dec 16 13:15:49 2016 Le Goff Kawrantin
** Last update Tue Dec 20 22:08:41 2016 Le Goff Kawrantin
*/

#include "include/my.h"

int	pos_O_in_tab(struct s_map *m, char **map, int lines, int cols)
{
  int	i;
  int	j;
  int	x;

  i = j = x = 0;
  if ((m->pos_O = malloc(sizeof(int *) * m->size_tab)) == NULL)
    exit(84);
  while (i < m->size_tab)
    ((m->pos_O[i++] = malloc(sizeof(int) * 2)) == NULL) ? exit(84) : 0;
  i = 0;
  while (j < lines)
    {
      if (map[j][i] == 'O')
	{
	  m->pos_O[x][0] = i;
	  m->pos_O[x++][1] = j;
	}
      if (i++ == cols)
	{
	  i = 0;
	  j++;
	}
    }
  return (0);
}

int    get_pos(char **map, int lines, int cols, struct s_map *m)
{
  int   i;
  int   j;

  i = -1;
  j = -1;
  while (++j < lines)
    {
      while (++i < cols)
	{
	  if (map[j][i] == 'P')
	    {
	      m->X = i;
	      m->Y = j;
	    }
	}
      i = 0;
    }
  pos_O_in_tab(m, map, lines, cols);
  return (0);
}
