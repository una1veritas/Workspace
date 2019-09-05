/*
** my_put_nbr_bin.c for my_put_nbr_bin in /home/kawrantin/delivery/PSU_2016_my_printf
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Tue Nov 15 21:52:58 2016 Le Goff Kawrantin
** Last update Wed Nov 16 18:49:34 2016 Le Goff Kawrantin
*/

#include <stdarg.h>
#include <stdlib.h>
#include "include/my.h"

int	final_bin(char *bin, int i, int bool, int size)
{
  if (bool == 42)
    {
      bin[i] = '-';
      bin = my_revstr(bin);
      my_putstr(bin);
    }
  else if (bool == 84)
    {
      bin[i] = '0';
      my_putstr(bin);
    }
  else
    {
      bin = my_revstr(bin);
      my_putstr(bin);
    }
  size = my_strlen(bin);
  return (size);
}

int	my_put_nbr_bin(va_list *ap)
{
  char		*bin;
  int		i;
  unsigned int	quotient;
  int		bool;
  int		size;

  bin = malloc(sizeof(char) * 100);
  quotient = va_arg(*ap, unsigned int);
  i = bool = 0;
  if (quotient == 0)
    bool = 84;
  while (quotient != 0)
    {
      bin[i] = (quotient % 2) + 48;
      quotient = quotient / 2;
      i++;
    }
  size = final_bin(bin, i, bool, size);
  return (size);
}
