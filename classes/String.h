#ifndef header_String
#define header_String

#include <iostream.h>
#include "Generic.h"

class String : public Generic {

  // class methods ;

 public:
  // constructor & destructor ;
  String(const char * = "");
  String(const String &);
  ~String();
  
  // instance variables ;
 private:
  char * string;
  int string_length;
  
  // instance methods ;
 public:
  
  // instance access ;
  //  const char * operator *() const;
  const int length() const;
  char & operator [] (const int);
  
  // conversion ;
  String & operator = (const String & str);
  String & operator = (const char *);
  String & operator += (const String & str);  String & operator += (const char *);
  String & operator + (const String &) const;
  String & operator + (const char *) const;

  String & copyFrom_to(int, int) const;

  // comparing ;
  virtual const unsigned long hash() const;
  virtual const Boolean isEqualTo(const Generic & ) const;

  // substring inspection ;
  int contains(const String &) const;
  int contains(const char *) const;

  // printing ;
  ostream & printOn(ostream & stream) const;

  // input & output ;
  friend istream & operator >> (istream &, String &);
};

#endif
