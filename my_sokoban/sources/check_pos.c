/*
** check_pos.c for  in /home/kawrantin/delivery/PSU_2016_my_sokoban
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Tue Dec 20 22:06:02 2016 Le Goff Kawrantin
** Last update Tue Dec 20 22:06:04 2016 Le Goff Kawrantin
*/

#include <include/my.h>

int     detect_gameover_trash(char **map, int i, int j, int a)
{
  if (map[j][i - 1] == '#' || map[j][i - 1] == 'X')
    {
      if (map[j - 1][i] == '#' || map[j - 1][i] == 'X')
	a = 1;
    }
  if (map[j - 1][i] == '#' || map[j - 1][i] == 'X')
    {
      if (map[j][i + 1] == '#' || map[j][i + 1] == 'X')
	a = 1;
    }
  if (map[j][i + 1] == '#' || map[j][i + 1] == 'X')
    {
      if (map[j + 1][i] == '#' || map[j + 1][i] == 'X')
	a = 1;
    }
  if (map[j + 1][i] == '#' || map[j + 1][i] == 'X')
    {
      if (map[j][i - 1] == '#' || map[j][i - 1] == 'X')
	a = 1;
    }
  return (a);
}

int     detect_gameover(char **map, int i, int j, struct s_map *m)
{
  int   a;

  a = 0;
  if (m->lines > 3 && m->cols > 3)
    a = detect_gameover_trash(map, i, j, a);
  return (a);
}

int     check_O(struct s_map *m, char **map)
{
  int   x;

  x = 0;
  while (x != m->size_tab)
    {
      if (map[m->pos_O[x][1]][m->pos_O[x][0]] != 'X')
	map[m->pos_O[x][1]][m->pos_O[x][0]] = 'O';
      x++;
    }
  return (0);
}

int     check_X(struct s_map *m, char **map)
{
  int   i;
  int   j;
  int   nbr_x;

  i = j = 0;
  nbr_x = m->size_tab;
  while (j < m->lines)
    {
      if (map[j][i] == 'X')
	{
	  nbr_x -= detect_gameover(map, i, j, m);
	}
      i++;
      if (i == m->cols)
	{
	  i = 0;
	  j++;
	}
    }
  if (nbr_x == 0)
    {
      endwin();
      exit(1);
    }
  return (0);
}
