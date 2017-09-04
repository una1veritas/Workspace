#ifndef header_Set
#define header_Set

#include <iostream.h>
#include "Collection.h"

class Set : public Collection {

 protected:
  // instance variables;
  const Generic ** elements;
  
 public:
  // constructor & destructor //
  Set(const unsigned long sz = 4);
  Set(const Set& );
  ~Set();

  // instance methods ;
  const unsigned long findIndex(const Generic& ) const;
  void grow(void);
  
 public:
  // accessing ;
  virtual const Generic& add(const Generic& );
  virtual void addAll(const Set& );
  virtual const Generic & remove(const Generic& anObj) {
    unsigned long i;
    const Generic * theObj;
    i = findIndex(anObj);
    if ((theObj = elements[i]) == NULL)
      return (*theObj);
    elements[i] = NULL;
    element_count--;
    return (*theObj);
  }
  
  virtual const Boolean removeAll(void);
  virtual const Boolean includes(const Generic& ) const;
  virtual const unsigned long size(void) const;
  virtual const Generic ** elementArray() {
    const Generic ** list;
    unsigned long i, j;
    list = new const Generic * [element_count+1];
    for (i = 0, j = 0; i < element_count; j++) {
      if (elements[j] != NULL) {
	list[i] = elements[j];
	i++;
      }
    }
    return list;
  }
  
  
  // comparing; 
  // virtual const unsigned long hash() const;
  virtual const Boolean isEqualTo(const Generic& ) const;
  
  // printing ;
  virtual ostream& printOn(ostream& stream) const {
    unsigned long i =0, cnt = 0;
    stream << " Set(";
    while (i < basic_size && cnt < MAX_PRINT_SIZE) {
      if (elements[i] != NULL) {
	stream << " ";
	stream << *(elements[i]);
	cnt++;
      }
      i++;
    }
    stream << ") ";
    return stream;
  }
  

};
#endif
