/*
** my_strncpy.c for my_strncpy in /home/kawrantin/delivery/CPool_Day06
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Mon Oct 10 15:19:33 2016 Le Goff Kawrantin
** Last update Mon Oct 10 18:16:57 2016 Le Goff Kawrantin
*/

char    *my_strncpy(char *dest, char *src, int n)
{
  int   i;
  int   k;
  int	a;
  int	b;

  a = 0;
  b = 0;
  k = 0;
  i = 0;
  while (src[i] != '\0')
    i++;
  if (n > i)
    n = i;
  while (k < n)
    {
      dest[k] = src[k];
      k++;
    }
  if (k == n)
    dest[n] = '\0';
  return (dest);
}
