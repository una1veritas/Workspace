/*
** my_put_nbr.c for my_put_nbr in /home/kawrantin/day3task7
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Tue Oct 11 09:34:29 2016 Le Goff Kawrantin
** Last update Tue Nov 29 09:23:46 2016 Le Goff Kawrantin
*/

#include "../../include/my.h"

int	my_put_nbr(int nb)
{
  int	modulo;
  int	size;

  size = 0;
  if (nb < 0)
    {
      my_putchar('-');
      nb = nb * (-1);
      size++;
    }
  if (nb >= 0)
    {
      if (nb >= 10)
	{
	  modulo = (nb % 10);
	  nb = (nb - modulo) / 10;
	  size++;
	  my_put_nbr(nb);
	  my_putchar(48 + modulo);
	}
      else
	my_putchar(48 + nb);
    }
  return (size);
}
