/*
** moves.c for  in /home/kawrantin/delivery/PSU_2016_my_sokoban
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Mon Dec 19 16:41:45 2016 Le Goff Kawrantin
** Last update Mon Dec 19 20:00:56 2016 Le Goff Kawrantin
*/

#include "include/my.h"

int	move_up(struct s_map *m, char **map)
{
  if (mvinch(m->Y - 1, m->X) == 'X' && mvinch(m->Y - 2, m->X) != '#')
    {
      if (mvinch(m->Y - 2, m->X) != 'X')
	{
	  map[m->Y - 2][m->X] = 'X';
	  map[m->Y - 1][m->X] = ' ';
	  m->Y = m->Y - 1;
	}
    }
  else if (mvinch(m->Y - 1, m->X) == ' ' || mvinch(m->Y - 1, m->X) == 'O')
    {
      m->Y = m->Y - 1;
    }
  return (0);
}

int	move_down(struct s_map *m, char **map)
{
  if (mvinch(m->Y + 1, m->X) == 'X' && mvinch(m->Y + 2, m->X) != '#')
    {
      if (mvinch(m->Y + 2, m->X) != 'X')
	{
	  map[m->Y + 2][m->X] = 'X';
	  map[m->Y + 1][m->X] = ' ';
	  m->Y = m->Y + 1;
	}
    }
  else if (mvinch(m->Y + 1, m->X) == ' ' || mvinch(m->Y + 1, m->X) == 'O')
    {
      m->Y = m->Y + 1;
    }
  return (0);
}

int	move_right(struct s_map *m, char **map)
{
  if (map[m->Y][m->X + 1] == 'X' && map[m->Y][m->X + 2] != '#')
    {
      if (map[m->Y][m->X + 2] != 'X')
	{
	  map[m->Y][m->X + 2] = 'X';
	  map[m->Y][m->X + 1] = ' ';
	  m->X = m->X + 1;
	}
    }
  else if (map[m->Y][m->X + 1] == ' ' || map[m->Y][m->X + 1] == 'O')
    {
      m->X = m->X + 1;
    }
  return (0);
}

int	move_left(struct s_map *m, char **map)
{
  if (mvinch(m->Y, m->X - 1) == 'X' && mvinch(m->Y, m->X - 2) != '#')
    {
      if (mvinch(m->Y, m->X - 2) != 'X')
	{
	  map[m->Y][m->X - 2] = 'X';
	  map[m->Y][m->X - 1] = ' ';
	  m->X = m->X - 1;
	}
    }
  else if (mvinch(m->Y, m->X - 1) == ' ' || mvinch(m->Y, m->X - 1) == 'O')
    {
      m->X = m->X - 1;
    }
  return (0);
}
