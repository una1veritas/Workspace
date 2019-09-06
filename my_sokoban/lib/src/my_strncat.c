/*
** my_strncat.c for my_strncat in /home/kawrantin/delivery/CPool_Day07
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Tue Oct 11 15:06:00 2016 Le Goff Kawrantin
** Last update Mon Oct 31 09:40:37 2016 Le Goff Kawrantin
*/

char    *my_strncat(char *dest, char *src, int nb)
{
  int   size_dest;
  int   size_src;
  int   k;
  int   l;

  size_dest = 0;
  size_src = 0;
  k = 0;
  l = 0;
  while (dest[size_dest] != '\0')
    size_dest++;
  while (src[size_src] != '\0')
    size_src++;
  if (nb > size_src)
    nb = size_src;

  while (k < nb)
    {
      dest[size_dest] = src[l];
      size_dest++;
      k++;
      l++;
    }
  return (dest);
}
