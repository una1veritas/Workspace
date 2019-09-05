/*
** open_map.c for open_map in /home/kawrantin/delivery/PSU_2016_my_sokoban
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Fri Dec  9 01:23:21 2016 Le Goff Kawrantin
** Last update Tue Dec 20 21:17:39 2016 Le Goff Kawrantin
*/

#include "include/my.h"

char	**buffer_to_map(char *buffer, int lines, int length)
{
  int	i;
  int	j;
  int	k;
  char	**map;

  i = j = k = 0;
  if ((map = malloc(sizeof(char *) * lines)) == NULL)
    exit(84);
  while (k < lines)
    {
      if ((map[k++] = malloc(sizeof(char) * (length + 1))) == NULL)
	exit(84);
    }
  k = -1;
  while (buffer[++k] != '\0')
    {
      if (buffer[k] == '\n')
	{
	  map[j++][i] = '\0';
	  i = 0;
	}
      else
	map[j][i++] = buffer[k];
    }
  return (map);
}

int	fill_init_map(char **map, struct s_map *m, int lines, int length)
{
  int	k;
  int	i;
  int	j;

  k = i = j = 0;
  if ((m->init_map = malloc(sizeof(char *) * lines)) == NULL)
    exit(84);
  while (k < lines)
    {
      if ((m->init_map[k++] = malloc(sizeof(char) * (length + 1))) == NULL)
	exit(84);
    }
  while (j < lines)
    {
      m->init_map[j][i] = map[j][i];
      i++;
      if (i == length)
	{
	  i = 0;
	  j++;
	}
    }
  return (0);
}

int	elem_error(char *buffer, struct s_map *m)
{
  int	i;
  int	nbr_P;
  int	nbr_O;
  int	nbr_X;

  i = nbr_P = nbr_O = nbr_X = 0;
  while (buffer[i] != '\0')
    {
      if (buffer[i] == 'P')
	nbr_P++;
      if (buffer[i] == 'O')
	nbr_O++;
      if (buffer[i] == 'X')
	nbr_X++;
      i++;
    }
  if (nbr_O < 1 || nbr_X < 1)
    exit(84);
  if (nbr_P != 1 || nbr_O != nbr_X)
    exit(84);
  m->size_tab = nbr_O;
  return (0);
}

char	**open_map(char *path, struct s_map *m)
{
  int	bytes;
  char	*buffer;
  char	**map;
  int	fd;
  int	lines;
  int	length;

  bytes = file_size(path);
  if ((buffer = malloc(sizeof(char) * (bytes + 1))) == NULL)
    exit(84);
  fd = open(path, O_RDONLY);
  bytes = read(fd, buffer, bytes);
  buffer[bytes] = '\0';
  close(fd);
  elem_error(buffer, m);
  lines = m->lines = nbr_lines(buffer);
  length = m->cols = max_length(buffer);
  map = buffer_to_map(buffer, lines, length);
  fill_init_map(map, m, lines, length);
  free(buffer);
  return (map);
}
