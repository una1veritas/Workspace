/*
** print_help.c for print_help in /home/kawrantin/delivery/PSU_2016_my_sokoban
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Fri Dec  9 01:07:38 2016 Le Goff Kawrantin
** Last update Tue Dec 20 10:11:04 2016 Le Goff Kawrantin
*/

#include "include/my.h"

int     print_help()
{
  my_printf(0, 0, "USAGE\n");
  my_printf(0, 0, "\t   ./my_sokoban map\n\n");
  my_printf(0, 0, "DESCRIPTION\n");
  my_printf(0, 0, "\t   map    file representing the warehouse map, ");
  my_printf(0, 0, "containing '#' for walls,\n");
  my_printf(0, 0, "\t\t  'P' for the player, 'X' for the boxes and 'O' for");
  my_printf(0, 0, " storage location.\n");
  return (0);
}
