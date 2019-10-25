/*
** map_measures.c for map_measures in /home/kawrantin/delivery/PSU_2016_my_sokoban
** 
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
** 
** Started on  Fri Dec  9 02:59:20 2016 Le Goff Kawrantin
** Last update Tue Dec 20 21:15:54 2016 Le Goff Kawrantin
*/

#include "include/my.h"

int     file_size(char *path)
{
  int   size;
  int   tmp;
  int   fd;
  char  *buff;

  buff = malloc(sizeof(char) * 10);
  fd = open(path, O_RDONLY);
  size = 10;
  tmp = read(fd, buff, 10);
  while (tmp == 10)
    {
      tmp = read(fd, buff, 10);
      size += 10;
    }
  close(fd);
  free(buff);
  return (size);
}

int     nbr_lines(char *buffer)
{
  int   i;
  int   nbr;

  i = nbr = 0;
  while (buffer[i] != '\0')
    {
      if (buffer[i] == '\n')
	nbr++;
      i++;
    }
  return (nbr);
}

int     max_length(char *buffer)
{
  int   max;
  int   tmp;
  int   j;
  int   k;

  max = tmp = j = k = 0;
  while (buffer[j] != '\0')
    {
      while (buffer[j] != '\n')
	{
	  j++;
	  tmp++;
	}
      if (max < tmp)
	max = tmp;
      tmp = 0;
      j++;
    }
  return (max);
}
