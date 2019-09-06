##
## Makefile for Makefile in /home/kawrantin/delivery/PSU_2016_my_printf_bootstrap
## 
## Made by Le Goff Kawrantin
## Login   <kawrantin@epitech.net>
## 
## Started on  Mon Nov  7 13:59:57 2016 Le Goff Kawrantin
## Last update Tue Nov 15 16:41:35 2016 Le Goff Kawrantin
##

#SRCS	=	*.c

FLAG	=	-I../include -Llib -lmy

NAME	=	libmy.a

all:	$(NAME)

$(NAME):	$(OBJ)
	gcc -c src/*.c
	ar rc libmy.a *.o

clean:
	rm -f *.o

fclean:	clean
	rm -f $(NAME)
	rm -f *~

re: fclean all

.PHONY: all clean fclean re
