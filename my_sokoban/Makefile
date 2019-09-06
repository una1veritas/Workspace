##
## Makefile for Makefile in /home/kawrantin/delivery/PSU_2016_my_printf_bootstrap
## 
## Made by Le Goff Kawrantin
## Login   <kawrantin@epitech.net>
## 
## Started on  Mon Nov  7 13:59:57 2016 Le Goff Kawrantin
## Last update Tue Dec 20 22:16:49 2016 Le Goff Kawrantin
##

SRCS	=	sources/main.c		\
		sources/print_help.c	\
		sources/open_map.c	\
		sources/map_measures.c	\
		sources/get_pos.c	\
		sources/check_pos.c	\
		sources/key_events.c	\
		sources/victory_error.c	\
		sources/moves.c		\
		sources/game.c

CC	=	gcc

CFLAGS	=	-Wall -Werror -Wextra -I.

LDFLAGS	=	-lncurses -Llib -lmy

OBJ	=	$(SRCS:.c=.o)

NAME	=	my_sokoban

all:	$(NAME)

lib/libmy.a: 
	cd lib/my; make; mv libmy.a ../libmy.a

$(NAME):	lib/libmy.a $(OBJ)
	$(CC) $(OBJ) -o $(NAME) $(LDFLAGS)

clean:
	rm -f $(OBJ)

fclean:	clean
	rm -f $(NAME)
	rm -f *~

re: fclean all

.PHONY: all clean fclean re
