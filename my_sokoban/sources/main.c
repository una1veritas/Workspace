/*
** main.c for main in /home/kawrantin/delivery/PSU_2016_my_sokoban
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Fri Dec  9 00:37:47 2016 Le Goff Kawrantin
** Last update Sun Dec 25 17:21:48 2016 Le Goff Kawrantin
*/

#include <stdlib.h>
#include "include/my.h"

int	main(int ac, char **av)
{
  struct s_map m;
  char	**map;

  if (ac == 2 && av[1][0] == '-' && av[1][1] == 'h')
    print_help();
  map = open_map(av[1], &m);
  game(map, &m);
  free(map);
  free(m.pos_O);
  free(m.init_map);
  return (0);
}
