/*
** my_printf.c for my_printf in /home/kawrantin/delivery/PSU_2016_my_printf
**
** Made by Le Goff Kawrantin
** Login   <kawrantin@epitech.net>
**
** Started on  Wed Nov  9 11:33:31 2016 Le Goff Kawrantin
** Last update Tue Dec  6 23:41:34 2016 Le Goff Kawrantin
*/

#include <stdarg.h>

#include "../src/include/my.h-"

int	tabb(int (*tab_fonc[256])())
{
  tab_fonc['d'] = &my_put_nbr_va;
  tab_fonc['i'] = &my_put_nbr_va;
  tab_fonc['s'] = &my_putstr_va;
  tab_fonc['S'] = &my_putstr_non_print;
  tab_fonc['u'] = &my_put_nbr_unsigned;
  tab_fonc['c'] = &my_putchar_va;
  tab_fonc['b'] = &my_put_nbr_bin;
  tab_fonc['o'] = &my_put_nbr_oct;
  tab_fonc['X'] = &my_put_nbr_hex_up;
  tab_fonc['x'] = &my_put_nbr_hex_low;
  tab_fonc['p'] = &my_put_address;
}

int	my_printf(int final_len, int i, char *format, ...)
{
  va_list	ap;
  int		(*tab_fonc[256])();

  va_start(ap, format);
  tabb(tab_fonc);
  while (format[i++] != '\0')
    {
      if ((format[i - 1] == '%') && (format[i] == '%'))
	{
	  final_len += the_little_mod_function_with_a_big_name(final_len);
	  i++;
	}
      else if (format[i - 1] == '%')
	{
	  final_len += tab_fonc[format[i]](&ap);
	  i++;
	}
      else
	{
	  my_putchar(format[i - 1]);
	  final_len++;
	}
    }
  va_end(ap);
  return (final_len);
}
