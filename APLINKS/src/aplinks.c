/***************************************************************

    aplinks ... Another pocket link file server Version 1.03

    (c)1992,93 N.Kon


    Usage: aplinks [ filename ... ]

    Last change: Jan. 8th, 1993

***************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef DOS_ORIGINAL
#include <dos.h>
#endif

#define SYSBUFSIZE 44*128
#define DIR_TOP 12*128

struct data_buffer {
    char    buffer[128];
};

char 	buffer[130];
char	system_buffer[SYSBUFSIZE + 1];
int	last_fat_number, current_dir;
FILE	*source_fd;
#ifdef DOS_ORIGINAL
struct data_buffer far *data_sector[980], far *temp_buffer_ptr;
#endif
struct data_buffer *data_sector[980], *temp_buffer_ptr;
char	filename[128];
struct find_t dos_file;

void epilogue(void);
char get_com(void);
void put_com(char);
void sread(int);
void swrite(int);
int check_com(void);
int check_ctrl_d(void);
void get_pathname(char * , char *);
void get_basename(char * , char *);
void get_extension(char * , char *);
void put_fat(int, int);
int get_fat(int);
void get_memory(void);
void take_file(void);
int find_first(void);
int find_next(void);

#ifdef DOS_ORIGINAL
void main(int argc, char *argv[])
#endif
int main(int argc, char *argv[])
{
    char	    sector_low, sector_high;
    int		    sector, i;
    int		    argn, found;

    printf("<<< Another pocket link file server for NEW-SC V1.03 (c)1992,93 N.Kon >>>\n\n");

    for( i = 0 ; i < SYSBUFSIZE ; i++ ) {
	system_buffer[i] = 0;
    }
    system_buffer[0] = 0xf0;

    for( i = 0 ; i <= 979 ; i++ ) {
	data_sector[i] = NULL;
    }

    argn = 1;
    current_dir = DIR_TOP;
    last_fat_number = 2;
    
    while ( argn < argc ) {
	strcpy( filename , argv[argn] );
	found = find_first();
	while ( found == NULL ) {
	    source_fd = fopen( filename , "rb" );
	    take_file();
	    fclose( source_fd );
	    found = find_next();
	}
	argn++;
    }

    printf("\nPlink server started.\n");

    do {
	printf( "\rO" );
	
	while( check_com() == EOF ) {
	    if ( check_ctrl_d() == NULL ) {
	        epilogue();
	    }
	}

	switch ( get_com() ) {
		case 'R':
		    printf("\rR");
		    sector_low = get_com();
		    sector_high = get_com();
		    sector = sector_low + sector_high * 256;
		    sread( sector );
		    for( i = 0; i <= 128 ; i++ ) {
			put_com( buffer[ i ] );
		    }
		    break;
		case 'W':
		    printf("\rW");
		    sector_low = get_com();
		    sector_high = get_com();
		    sector = sector_low + sector_high * 256;
		    for( i=0 ; i <= 129 ; i++ ) {
			buffer[i] = get_com();
		    }
		    if ( (buffer[129] & 0xff) == 0xff ) {
			swrite( sector );
			put_com( 0 );
		    }
		    break;
		case 'D':
		    epilogue();
	}
    } while (1);
}

void epilogue(void)
/* Terminate plink server and flush ordinaly files */
{
	    int	    i, j, fat_number, current_dir, condition;
	    char    filename[256];
	    long    file_size;

	    printf( "\r*\nPlink server halted. Syncing...\n" );
	    current_dir = DIR_TOP;
	    while ( current_dir < SYSBUFSIZE ) {
		if ( ( system_buffer[ current_dir ] != 0 ) && ( system_buffer[ current_dir ] != 0xe5 ) && ( system_buffer[ current_dir + 0x0b ] == 0x20 ) ) {
		    strcpy( filename , "" );
		    for( i = 0 ; i <= 7 ; i++ ) {
			if ( system_buffer[ current_dir + i ] != ' ' ) {
			    sprintf( filename , "%s%c" , filename , system_buffer[ current_dir + i ] );
			} else {
			    break;
			}
		    }
		    strcat( filename , "." );
		    for( i = 0 ; i <= 2 ; i++ ) {
			if ( system_buffer[ current_dir + i + 8 ] != ' ' ) {
			    sprintf( filename , "%s%c" , filename , system_buffer[ current_dir + i + 8 ] );
			} else {
			    break;
			}
		    }

		    file_size = system_buffer[ current_dir + 0x1c ];
		    file_size += system_buffer[ current_dir + 0x1d ] * 0x100;
		    file_size += system_buffer[ current_dir + 0x1e ] * 0x10000;

		    if ( ( source_fd = fopen( filename , "wb" ) ) == NULL ) {
			printf("Can\'t open \"%s\". Skip this file\n" , filename );
		    } else {
			printf( "Writing \"%s\" ... ", filename );
			condition = NULL;
			fat_number = system_buffer[ current_dir + 0x1a ];
			fat_number += system_buffer[ current_dir + 0x1b ] * 0x100;
			do {
			    j = 128;
			    if ( file_size < 128 ) {
				j = file_size;
			    }
			    for ( i = 1 ; i <= j ; i++ ) {
				condition |= fputc( data_sector[ fat_number - 2 ] -> buffer[ i - 1 ] , source_fd );
			    }
			    file_size -= 128;
			    fat_number = get_fat( fat_number );
			} while ( fat_number != 0xff0 );
			fclose( source_fd );
			if ( condition != EOF ) {
			    printf("done\n");
			} else {
			    printf("disk full. Skip this file\n");
			    remove( filename );
			}
		    }
		}
		current_dir += 32;
	    }
	    printf("done.\n");
	    exit(0);	
}

void sread(int sector)
/* Read datas from virtual disk and set it to "BUFFER" */
{
    int	    i;

    if ( sector >= 44 ) {
	if ( data_sector[ sector - 44 ] == NULL ) {
	    get_memory();
	    data_sector[ sector - 44 ] = temp_buffer_ptr;
	}
	for( i = 0 ; i <= 127 ; i++ ) {
	    buffer[i] = data_sector[ sector - 44 ] -> buffer[i];
	}
    } else {
	for( i = 0 ; i <= 128 ; i++ ) {
	    buffer[i] = system_buffer[sector * 128 + i];
	}
    }
}

void swrite(int sector)
/* Write datas of "BUFFER" to the virtual disk */
{
    int	    i;

    if ( sector >= 44 ) {
	if ( data_sector[ sector - 44 ] == NULL ) {
	    get_memory();
	    data_sector[ sector - 44 ] = temp_buffer_ptr;
	}
	for( i = 0 ; i <= 127 ; i++ ) {
	    data_sector[ sector - 44 ] -> buffer[i] = buffer[i];
	}
    } else {
	for( i = 0 ; i <= 128 ; i++ ) {
	    system_buffer[sector * 128 + i] = buffer[i];
	}
    }
}

char get_com(void)
/* Get a charactor from the serial port */
{
    union REGS    reg;

    reg.h.ah = 0x03;
    intdos( &reg , &reg );
    return( reg.h.al );
}

void put_com(char c)
/* Put a charactor to the serial port */
{
    union REGS    reg;

    reg.h.ah = 0x04;
    reg.h.dl = c;
    intdos( &reg , &reg );
}

int check_com()
/* if some data exist in serial port, returns NULL. Otherwise EOF */
{
    union REGS	  reg;

    reg.h.ah = 0x44;
    reg.h.al = 0x06;
    reg.x.bx = 0x03; /* Serial port */
    intdos( &reg , &reg );
    if ( reg.h.al == 0 ) {
	return( EOF );
    } else {
	return( NULL );
    }
}

int check_ctrl_d()
/* if pressed [CTRL]+[D], returns NULL. Otherwise EOF */
{
    union REGS	  reg;

    reg.h.ah = 0x44;
    reg.h.al = 0x06;
    reg.x.bx = 0x00; /* stdin */
    intdos( &reg , &reg );
    if ( reg.h.al == 0 ) {
	return( EOF );
    } else {
	reg.h.ah = 0x07;
	intdos( &reg , &reg );
	if ( reg.h.al == 0x04 ) {
	    return( NULL );
	} else {
	    return( EOF );
	}
    }
}

void get_pathname( char *pathname , char *filename )
/* Set the pathname of "filename" to "pathname" */
{
    int	    i, max;

    strcpy( pathname , "" );
    max = -1;
    for( i = 0 ; i < strlen( filename ) ; i++ ) {
	if ( ( filename[i] == ':' ) || ( filename[i] == '\\' ) ) {
	    max = i;
	}
    }

    for( i = 0 ; i <= max ; i++ ) {
	sprintf( pathname , "%s%c" , pathname , filename[i] );
    }
}

void get_basename( char *basename , char *filename )
/* Set the basename of "filename" to "basename" */
{
    char    temp_str[256];
    int	    i;

    strcpy( temp_str , filename );
    if ( strrchr( temp_str , ':' ) != NULL ) {
	strcpy( temp_str , strrchr( temp_str , ':' ) + 1 );
    }
    if ( strrchr( temp_str , '\\' ) != NULL ) {
	strcpy( temp_str , strrchr( temp_str , '\\' ) + 1 );
    }
    strcpy( basename , "" );
    for( i = 0 ; (i <= 7) && (temp_str[i] != NULL) ; i++ ) {
	if ( temp_str[i] != '.' ) {
	    sprintf( basename , "%s%c" , basename , temp_str[i] );
	} else {
	    break;
	}
    }
    strcat( basename , "        " );
}

void get_extension( char *extension , char *filename )
/* Set the extension of "filename" to "extension" */
{
    char    temp_str[256];

    strcpy( temp_str , filename );
    if ( strrchr( temp_str , ':' ) != NULL ) {
	strcpy( temp_str , strrchr( temp_str , ':' ) + 1 );
    }
    if ( strrchr( temp_str , '\\' ) != NULL ) {
	strcpy( temp_str , strrchr( temp_str , '\\' ) + 1 );
    }
    if ( strrchr( temp_str , '.' ) != NULL ) {
	strcpy( temp_str , strrchr( temp_str , '.' ) + 1 );
    } else {
	strcpy( temp_str , "" );
    }
    strcpy( extension , temp_str );
    strcat( extension , "        " );
}

void put_fat( int fat_number , int value )
/* Write a fat data "value" to where "fat_number" indicates */
{
    fat_number *= 3;
    if ( ( fat_number % 2 ) == 0 ) {
	system_buffer[ fat_number / 2 ] = value & 0xff;
	system_buffer[ fat_number / 2 + 1 ] = ( system_buffer[ fat_number / 2 + 1] & 0xf0 ) + value / 256;
    } else {
	system_buffer[ fat_number / 2 ] = ( system_buffer[ fat_number / 2 ] & 0xf ) + ( value & 0x0f ) * 16;
	system_buffer[ fat_number / 2 + 1 ] = value / 16;
    }
}

int get_fat( int fat_number )
/* Read a fat data from where "fat_number" indicates, and return it */
{
    int	    value;
    
    fat_number *= 3;
    if ( ( fat_number % 2 ) == 0 ) {
	value = system_buffer[ fat_number / 2 ];
	value += ( system_buffer[ fat_number / 2 + 1 ] & 0xf ) * 0x100;
    } else {
	value = system_buffer[ fat_number / 2 ] / 16;
	value += system_buffer[ fat_number / 2 + 1 ] * 16;
    }
    return( value );
}

void get_memory()
/* Get 128bytes from main-memory and set the pointer */
{
    temp_buffer_ptr = farmalloc( sizeof( struct data_buffer ) );
    if ( temp_buffer_ptr == NULL ) {
	printf( "\nCan\'t get enough memory\n" );
	exit(1);
    }
}

void take_file()
/* Take the DOS files into virtual disk */
{
    long    file_size;
    int	    i, read_count;
    char    basename[32], extension[32];
    char    first_sector;

    printf( "Reading \"%s\" ... " , filename );

    if ( current_dir >= SYSBUFSIZE ) {
	printf( "Exceeded 128 directory. Skip this file\n");
	return;
    }

    fseek( source_fd , 0 , SEEK_END );
    file_size = ftell( source_fd );
    fseek( source_fd , 0 , SEEK_SET );

    if ( ( ( file_size / 128 ) + 1 ) <= ( 982 - last_fat_number ) ) {
	get_basename( basename , filename );
	get_extension( extension , filename );
	
	for ( i = 0 ; i <= 7 ; i++ ) {
	    system_buffer[ current_dir + i ] = toupper( basename[ i ] );
	}
	for ( i = 0 ; i <= 2 ; i++ ) {
	    system_buffer[ current_dir + i + 8 ] = toupper( extension[ i ] );
	}
	system_buffer[ current_dir + 0x0b ] = 0x21;
	system_buffer[ current_dir + 0x1a ] = last_fat_number & 0xff;
	system_buffer[ current_dir + 0x1b ] = last_fat_number / 256;
	system_buffer[ current_dir + 0x1c ] = file_size & 0xff;
	system_buffer[ current_dir + 0x1d ] = ( file_size / 0x100 ) & 0xff;
	system_buffer[ current_dir + 0x1e ] = ( file_size / 0x10000 ) & 0xff;
	first_sector = !NULL;

	while ( ( read_count = fread( buffer , sizeof( char ) , 128 , source_fd ) ) == 128 ) {
	    get_memory();
	    data_sector[ last_fat_number - 2 ] = temp_buffer_ptr;
	    for( i = 0 ; i < read_count ; i++ ) {
		temp_buffer_ptr -> buffer[i] = buffer[i];
	    }
	    if ( first_sector == NULL ) {
		put_fat( last_fat_number - 1 , last_fat_number );
	    } else {
		first_sector = NULL;
	    }
	    last_fat_number++;
	}
	get_memory();
	data_sector[ last_fat_number - 2 ] = temp_buffer_ptr;
	for( i = 0 ; i < read_count ; i++ ) {
	    temp_buffer_ptr -> buffer[i] = buffer[i];
	}
	if ( first_sector == NULL ) {
	    put_fat( last_fat_number - 1 , last_fat_number );
	} else {
	    first_sector = NULL;
	}
	put_fat( last_fat_number , 0xff0 );
	last_fat_number++;
	current_dir += 32;
	printf( "done\n" , filename );
    } else {
	printf( "not enough disk. Skip this file\n" );
    }
}

int find_first()
/* Find a file which matches to the pattern "filename" first, and set it to "filename" */
{
    int	    found;
    char    pathname[128];
    
    found = _dos_findfirst( filename , _A_NORMAL , &dos_file );
    if ( found == NULL ) {
	get_pathname( pathname , filename );
	sprintf( filename , "%s%s" , pathname , dos_file.name );
    }
    return( found );
}

int find_next()
/* Find a file which matches to the pattern "filename" next, and set it to "filename" */
{
    int	    found;
    char    pathname[128];
    
    found = _dos_findnext( &dos_file );
    if ( found == NULL ) {
	get_pathname( pathname , filename );
	sprintf( filename , "%s%s" , pathname , dos_file.name );
    }
    return( found );
}
