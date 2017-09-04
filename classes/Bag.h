#ifndef header_Bag
#define header_Bag

#include <iostream.h>
#include "Generic.h"
#include "Set.h"

class Bag : public Set {

 protected:
  // class constants ;
  static unsigned long max_print_size;

 public:
  // constructor & destructor ;
  Bag(unsigned long sz);
  Bag(Bag& );
  ~Bag();

 protected:
  // instance variables ;
  // const Generic ** elements;
  // unsigned long count;
  unsigned long * occurrences;
  unsigned long total;

  // instance methods ;
  // const unsigned long findIndex(const Generic& ) const;
  void grow(void);

 public:
  // accessing ;
  virtual const Generic& add(const Generic& );
  virtual const Generic& add(const Generic&, const unsigned long);
  virtual const Generic & remove(const Generic& );
  virtual const Boolean remove(const Generic&, const unsigned long);
  // virtual const Boolean includes(const Generic& ) const;
  virtual const unsigned long occurrencesOf(const Generic& ) const;
  virtual const unsigned long size(void) const;

  // comparing ;
  //virtual const unsigned long hash() const;
  virtual const Boolean isEqualTo(const Generic& ) const;

  // printing ;
  virtual ostream& printOn(ostream& stream) const;
};
#endif
