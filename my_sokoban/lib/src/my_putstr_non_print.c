/*
** my_putstr_non_pint.c for my_putstr_non_print in /home/kawrantin/delivery/PSU_2016_my_printf
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Fri Nov 18 14:51:39 2016 Le Goff Kawrantin
** Last update Thu Dec  1 09:09:26 2016 Le Goff Kawrantin
*/

#include <stdlib.h>

#include "../src/include/my.h-"

int	my_putstr_non_print(va_list *ap)
{
  int	i;
  char	*str;
  int	size;

  i = 0;
  str = va_arg(*ap, char*);
  size = my_strlen(str);
  while (str[i] != '\0')
    {
      if (str[i] <= 31)
	{

	}
      my_putchar(str[i]);
      i++;
    }
  return (size);
}
