/*
** my_strcat.c for my_strcat in /home/kawrantin/delivery/CPool_Day07
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Tue Oct 11 13:53:25 2016 Le Goff Kawrantin
** Last update Mon Oct 31 09:40:06 2016 Le Goff Kawrantin
*/

char	*my_strcat(char *dest, char *src)
{
  int	size_dest;
  int	size_src;
  int	k;
  int	l;

  size_dest = 0;
  size_src = 0;
  k = 0;
  l = 0;
  while (dest[size_dest] != '\0')
    size_dest++;
  while (src[size_src] != '\0')
    size_src++;
  while (k < size_src)
    {
      dest[size_dest] = src[l];
      size_dest++;
      k++;
      l++;
    }
  return (dest);
}
