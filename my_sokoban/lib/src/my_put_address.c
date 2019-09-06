/*
** truc.c for my_put_adress in /home/kawrantin/delivery/PSU_2016_my_printf
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Thu Nov 17 15:38:39 2016 Le Goff Kawrantin
** Last update Thu Dec  1 09:09:01 2016 Le Goff Kawrantin
*/

#include <stdlib.h>

#include "../src/include/my.h-"

int     final_ptr(char *bin, int i, int bool, int size)
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

int     my_put_nbr_ptr(long tmp)
{
  char          *bin;
  int           i;
  long	int	quotient;
  int           bool;
  int           size;

  bin = malloc(sizeof(char) * 100);
  quotient = tmp;
  i = bool = 0;
  if (quotient == 0)
    bool = 84;
  while (quotient != 0)
    {
      if ((quotient % 16) >= 10)
	bin[i] = (quotient % 16) + 87;
      else
	bin[i] = (quotient % 16) + 48;
      quotient = quotient / 16;
      i++;
    }
  size = final_ptr(bin, i, bool, size);
  return (size);
}

int	my_put_address(va_list *ap)
{
  int	size;
  char	*address;
  void	*tmp;

  tmp = va_arg(*ap, void*);
  my_putstr("0x");
  my_put_nbr_ptr((long)tmp);
  return (size);
}
