/*
** my_putchar_va.c for my_putchar_va in /home/kawrantin/delivery/PSU_2016_my_printf
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Tue Nov 15 21:26:02 2016 Le Goff Kawrantin
** Last update Wed Nov 16 18:54:35 2016 Le Goff Kawrantin
*/

#include <unistd.h>
#include <stdarg.h>
#include "include/my.h"

int	my_putchar_va(va_list *ap)
{
  int	size;
  char	c;

  size = 1;
  c = va_arg(*ap, int);
  write(1, &c, 1);
  return (size);
}
