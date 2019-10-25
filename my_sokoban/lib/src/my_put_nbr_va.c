/*
** my_put_nbr.c for my_put_nbr_va in /home/kawrantin/delivery/PSU_2016_my_printf
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Tue Nov 15 19:28:53 2016 Le Goff Kawrantin
** Last update Wed Nov 16 18:55:39 2016 Le Goff Kawrantin
*/

#include <stdarg.h>

#include "../src/include/my.h-"

int	disp(int nb, int size)
{
  int	modulo;

  if (nb >= 0)
    {
      if (nb >= 10)
	{
	  modulo = (nb % 10);
	  nb = (nb - modulo) / 10;
	  size = disp(nb, size);
	  my_putchar(48 + modulo );
	}
      else
	my_putchar(48 + nb);
    }
  size++;
  return (size);
}

int	my_put_nbr_va(va_list *ap)
{
  int	size;
  int	nb;

  size = 0;
  nb = va_arg(*ap, int);
  if (nb < 0)
    {
      my_putchar('-');
      nb = nb * (-1);
      size++;
    }
  size = disp(nb, size);
  return (size);
}
