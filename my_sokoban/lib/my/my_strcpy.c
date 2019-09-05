/*
** my_strcpy.c for my_strcpy in /home/kawrantin/delivery/CPool_Day06
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Mon Oct 10 10:43:59 2016 Le Goff Kawrantin
** Last update Mon Oct 10 15:15:38 2016 Le Goff Kawrantin
*/
char	*my_strcpy(char *dest, char *src)
{
  int	i;
  int	k;

  k = 0;
  i = 0;
  while (src[i] != '\0')
    {
      i++;
    }
  while (k < i + 1)
    {
      dest[k] = src[k];
      k++;
    }
  return (dest);
}
