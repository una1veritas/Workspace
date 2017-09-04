#include <string.h>
#include "String.h"

/*
  constructor & destructor */

/*
 String::String() {
   string = new char[32];
   basic_size = 32;
   string_length = 0;
   *string = '\0';
 }
*/

 String::String(const char * s) {
   string_length = strlen(s);
   basic_size = ((string_length < 31)? 31 : string_length) + 1;
   string = new char[basic_size];
   strcpy(string,s);
 }

 String::String(const String & s) {
   string = new char[s.string_length+1];
   basic_size = s.basic_size;
   string_length = s.string_length;
   strcpy(string,s.string);
 }

 String::~String() {
   delete [] string;
 }

/*
  instance access */
// const char * String::operator *() const {
//  return string; 
// }

char & String::operator [] (const int i) {
  if (i < 0 || i > string_length) {
    error ("Error: String operator [] :index out of bounds\n");
  } else {
    return string[i];
  }
}

String & String::operator = (const String & s) {
  return *this = s.string;
}

String & String::operator = (const char * s) {
  if ((string_length = strlen(s)) > basic_size) {
    delete string;
    string = new char[string_length+1];
    basic_size = string_length + 1;
  }
  strcpy(string, s);
  return *this;
}

String & String::operator += (const String & s) {
  return *this += s.string;
}

String & String::operator += (const char * s) {
  int l;
  char * newstr;

  if ((l = string_length + strlen(s)) > basic_size) {
    newstr = new char[l+1];
    basic_size = l+1;
    string_length = l;
    strcpy(newstr, string);
    strcat(newstr, s);
    delete string;
    string = newstr;
  } else {
    string_length = l;
    strcat(string, s);
  }
  return *this;
}

String & String::copyFrom_to(int b, int e) const {
  String * ns;

  if (b < 0 || e > string_length || e - b < 0) {
    error("Error: String copyFrom_to :index out of bounds\n");
  } else {
    ns = new String(new char [e - b + 1]);
    strncpy(ns->string, string+b, e - b + 1);
    *(ns->string+e-b+1) = '\0';
  }
  return *ns;
}

const int String::length() const {
  return string_length;
}

const Boolean String::isEqualTo(const Generic & s) const {
  //cerr << " (String::isEqualTo) ";
  return (Boolean) (strcmp(string, ((String&) s).string) == 0);
}

const unsigned long String::hash() const {
  /*
    size of long == 4
   */
  unsigned long midLeft, midRight, firstc, mlc, mrc, nlc, lastc;

  if (string_length == 0) {
    return 1234;
  } else {
    if (string_length == 1) {
      midLeft = midRight = 1;
    } else {
      midLeft = string_length >> 1;
      midRight = midLeft + 1;
    }
  }
  firstc = *string;
  mlc = string[midLeft-1];
  mrc = string[midRight-1];
  nlc = string[string_length-2];
  lastc = string[string_length-1];
  
  return (mrc + 
	  (((mrc + string_length) & 0xffff) << 8) + 
	  (((firstc + nlc) & 0x3fff) << 12) + 
	  (nlc << 2) +
	  ((lastc & 0x07ff) << 16) + ((firstc + lastc) << 4) +
	  ((mlc & 0xff) << 20) + (mlc << 6));
}

int String::contains(const String & s) const {
  return strstr(string, s.string) != NULL;
}

int String::contains(const char * s) const {
  return strstr(string, s) != NULL;
}

String & String::operator + (const String & s2) const {
  return *(new String(string)) += s2;
}

String & String::operator + (const char * s2) const {
  return *(new String(string)) += s2;
}

ostream & String::printOn(ostream & stream) const {
  return stream << "String(\"" << string << "\")";
}

istream & operator >> (istream & s, String & str) {
  return s.get(str.string, (int) str.basic_size, '\n');
}
