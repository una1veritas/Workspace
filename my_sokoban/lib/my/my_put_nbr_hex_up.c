/*
** my_put_nbr_hex.c for my_put_nbr_hex_up in /home/kawrantin/delivery/PSU_2016_my_printf
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Thu Nov 17 12:55:41 2016 Le Goff Kawrantin
** Last update Thu Nov 17 12:55:44 2016 Le Goff Kawrantin
*/

#include <stdarg.h>
#include <stdlib.h>
#include "include/my.h"

int	final_hex_up(char *bin, int i, int bool, int size)
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

int	my_put_nbr_hex_up(va_list *ap)
{
  char		*bin;
  int		i;
  unsigned int  quotient;
  int		bool;
  int		size;

  bin = malloc(sizeof(char) * 100);
  quotient = va_arg(*ap, unsigned int);
  i = bool = 0;
  if (quotient == 0)
    bool = 84;
  while (quotient != 0)
    {
      if ((quotient % 16) >= 10)
	bin[i] = (quotient % 16) + 55;
      else
	bin[i] = (quotient % 16) + 48;
      quotient = quotient / 16;
      i++;
    }
  size = final_hex_up(bin, i, bool, size);
  return (size);
}
