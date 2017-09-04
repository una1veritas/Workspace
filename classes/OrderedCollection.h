#ifndef header_OrderedCollection
#define header_OrderedCollection

#include <Collection.h>

class OrderedCollection : public Collection {

 private:
  const Generic ** elements;

 public:
  OrderedCollection(int sz = 4) {
    unsigned long i;
    //cerr << "Called constructor.\n";
    //cerr.flush();
    size_limit = (sz > 4)? sz : 4;
    element_count = 0;
    elements = new const Generic * [size_limit];
    for (i=0; i<size_limit; i++) {
      elements[i] = NULL;
    }
    //cerr << "Finished constructor.\n";
    //cerr.flush();

  }

  ~OrderedCollection() {
    delete elements;
  }

  const Generic & operator [] (const unsigned long i) {
    if ((i < 0) || (i >= element_count)) 
      error ("Error: OrderedCollection operator [] :index out of bounds\n");
    return *elements[i];
  }

  const Generic & last(void) {
    return *elements[element_count];
  }

  unsigned long size(void) {
    return element_count;
  }

  virtual const Generic& addLast(const Generic& obj) {
    if (! (element_count < size_limit))
      this->grow();
    elements[element_count] = & obj;
    element_count++;
    return obj;
  }


  // virtual const Boolean remove(const Generic& );

  virtual const Boolean includes(const Generic& obj) const {
    unsigned long i;
    for (i = 0; i < element_count; i++) {
      if ( *(elements[i]) == obj )
	return true;
    }
    return false;
  }

  void grow(void) {
    const Generic ** array;
    unsigned long i, newlimit;

    //cerr << "Called grow.\n";
    //cerr.flush();
    newlimit = element_count + (element_count/2) + 1;
    array = new const Generic * [newlimit];
    for (i = 0; i < element_count; i++) {
      array[i] = elements[i];
    }
    for (i = element_count; i < size_limit; i++) {
      array[i] = NULL;
    }
    delete elements;
    elements = array;
    size_limit = newlimit;
    return;
  }
  
  virtual const unsigned long size(void) const {
    return element_count;
  }

  ostream & printOn(ostream & stream) const {
    unsigned long i;
    stream << " OrderedCollection(";
    i = 0;
    while (i < element_count ) {
      stream << " ";
      stream << *(elements[i]);
      i++;
    }
    stream << ") ";
    return stream;
  }


};
#endif
