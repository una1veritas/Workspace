/*
** my_revstr.c for my_revstr in /home/kawrantin/delivery/CPool_Day06
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Mon Oct 10 19:18:36 2016 Le Goff Kawrantin
** Last update Mon Oct 10 22:40:45 2016 Le Goff Kawrantin
*/

char	*my_revstr(char *str)
{
  int	i;
  int	y;
  char	tmp;

  i = 0;
  y = 0;
  tmp = 0;
  while (str[y] != '\0')
    y++;
  y--;
  while (i < y)
    {
      tmp = str[i];
      str[i] = str[y];
      str[y] = tmp;
      i++;
      y--;
    }
  return (str);
}
