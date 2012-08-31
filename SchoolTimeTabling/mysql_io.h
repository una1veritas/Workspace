typedef struct association {
  char * label;
  union {
    int integer;
    char * string;
  };
} association;

int read_from_db(char * host, char * user, char * passwd, char * db_name, 
		 char * label_field, char * value_field, char * table_name,
		 association assoc[]);

char * get_value_from_assoc(association assoc[], char * label);
