/*
** my_strlen.c for my_strlen in /home/kawrantin/delivery/CPool_Day04
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Fri Oct  7 22:25:56 2016 Le Goff Kawrantin
** Last update Fri Oct  7 23:37:12 2016 Le Goff Kawrantin
*/

int	my_strlen(char *str)
{
  int i;

  i = 0;
  while (str[i] != '\0')
    {
      i++;
    }
  return (i);
}
