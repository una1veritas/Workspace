/*
** my_putstr_va.c for my_putstr_va in /home/kawrantin/delivery/PSU_2016_my_printf
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Tue Nov 15 20:00:20 2016 Le Goff Kawrantin
** Last update Sat Nov 19 13:45:46 2016 Le Goff Kawrantin
*/

#include <stdlib.h>
#include "include/my.h"

int	my_putstr_va(va_list *ap)
{
  int	i;
  char	*str;
  int	size;

  i = 0;
  str = va_arg(*ap, char*);
  size = my_strlen(str);
  while (str[i] != '\0')
    {
      my_putchar(str[i]);
      i++;
    }
  return (size);
}
