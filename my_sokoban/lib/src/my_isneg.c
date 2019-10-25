/*
** my_isneg.c for my_isneg in /home/kawrantin/test
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Wed Oct  5 13:21:50 2016 Le Goff Kawrantin
** Last update Wed Nov 16 23:04:35 2016 Le Goff Kawrantin
*/

#include "../../include/my.h"

int	my_isneg(int n)
{
  if (n < 0)
    {
      my_putchar(78);
    }
  else
    {
      my_putchar(80);
    }
  return (0);
}
