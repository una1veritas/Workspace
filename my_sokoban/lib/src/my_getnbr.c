/*
** my_getnbr.c for my_getnbr in /home/kawrantin
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Tue Oct 11 10:42:50 2016 Le Goff Kawrantin
** Last update Mon Oct 31 09:38:33 2016 Le Goff Kawrantin
*/

int	my_getnbr(char *str)
{
  int	nbr;
  int	i;
  int	neg;

  nbr = 0;
  i = 0;
  neg = 0;
  while (str[i] != '\0')
    {
      if ((nbr >= 2147483647) || (nbr <= -2147483647))
	return (0);
      if ((str[i] > 47) && (str[i] < 58))
	{
	  if (str[i - 1] == '-')
	    neg = 1;
	  nbr += (str[i] - 48);
	  if ((str[i + 1] > 47) && (str[i + 1] < 58))
	    nbr = nbr * 10;
	  else if (neg == 1)
	    return (-nbr);
	  else
	    return (nbr);
	}
      i++;
    }
}
