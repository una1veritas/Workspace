/*
** my_putstr.c for my_putstr in /home/kawrantin/delivery/CPool_Day04
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Fri Oct  7 21:08:40 2016 Le Goff Kawrantin
** Last update Tue Nov 29 13:18:42 2016 Le Goff Kawrantin
*/

#include "../../include/my.h"

int	my_putstr(char *str)
{
  int i;

  i = 0;
  while (str[i] != '\0')
    {
      my_putchar(str[i]);
      i++;
    }
}
