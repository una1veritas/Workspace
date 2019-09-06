/*
** my_swap.c for my_swap in /home/kawrantin/delivery/CPool_Day03
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Fri Oct  7 12:50:53 2016 Le Goff Kawrantin
** Last update Fri Oct  7 23:24:55 2016 Le Goff Kawrantin
*/

int	my_swap(int *a, int *b)
{
  int c;

  c = *a;
  *a = *b;
  *b = c;
}
