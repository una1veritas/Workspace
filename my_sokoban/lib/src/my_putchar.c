/*
** my_putchar.c for my_putchar in /home/kawrantin/librairy
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Tue Oct 11 09:04:33 2016 Le Goff Kawrantin
** Last update Wed Nov 16 23:00:39 2016 Le Goff Kawrantin
*/

#include <unistd.h>

void	my_putchar(char c)
{
  write(1, &c, 1);
}
