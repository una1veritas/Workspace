/*
** my_put_nbr_unsigned.c for my_put_nbr_unsigned in /home/kawrantin/delivery/PSU_2016_my_printf
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Tue Nov 15 18:15:20 2016 Le Goff Kawrantin
** Last update Wed Nov 16 18:56:58 2016 Le Goff Kawrantin
*/

#include <stdarg.h>
#include "include/my.h"

int     disp_unsigned(unsigned int nb, int size)
{
  int	modulo;

  if (nb >= 0)
    {
      if (nb >= 10)
	{
	  modulo = (nb % 10);
	  nb = (nb - modulo) / 10;
	  size = disp_unsigned(nb, size);
	  my_putchar(48 + modulo );
	}
      else
	my_putchar(48 + nb);
    }
  size++;
  return (size);
}

int     my_put_nbr_unsigned(va_list *ap)
{
  int		size;
  unsigned int	nb;

  size = 0;
  nb = va_arg(*ap, unsigned int);
  size = disp_unsigned(nb, size);
  return (size);
}
