#include <string.h>
#include <stdlib.h>

#include <sys/time.h>
#include <stdio.h>
#include <mysql/mysql.h>

#include "mysql_io.h"

int read_from_db(char * host, char * user, char * passwd, char * db_name, 
		 char * label_field, char * value_field, char * table_name,
		 association assoc[]) {
  MYSQL_RES * result;
  MYSQL_ROW row;
  MYSQL * conn, mysql;
  MYSQL_FIELD * fields;

  char sql[256];
  //int state;
  //int i, num_o_fields;
  int j;

  mysql_init(&mysql);
  conn = mysql_real_connect(&mysql, host, user, passwd, db_name, 0, "", 0);
  if (conn == NULL) {
    printf("%s\n", mysql_error(&mysql));
    return 1;
  }

  sprintf(sql,"SELECT %s, %s FROM %s",label_field, value_field, table_name);
  if (mysql_query(conn, sql) != 0) {
    printf("%s\n", mysql_error(conn));
    return 1;
  }

  result = mysql_store_result(conn);
//  printf("Rows: %d\n", mysql_num_rows(result)); 
//  printf("Fields totally: %d\n", 
//	 num_o_fields = mysql_num_fields(result));
  
  fields = mysql_fetch_field(result);

  while ( (row = mysql_fetch_row(result)) != NULL ) {
    for (j = 0; assoc[j].label != NULL; j++) {
      if ( strcmp(row[0],assoc[j].label) == 0) {
	if (IS_NUM(fields[1].type)) {
	  assoc[j].integer = atoi(row[1]);
	} else {
	  strcpy(row[1],assoc[j].string);
	}
      }
    }
  }

  mysql_free_result(result);

  mysql_close(conn);
  
//  printf("\nFinished.\n");
  return 0;
}

char * get_value_from_assoc(association assoc[], char * label) {
  int i;
  for (i = 0; assoc[i].label != NULL; i++) {
    if ( strcmp(label, assoc[i].label) == 0 ){
      return assoc[i].string;
    }
  }
  return 0;
}

