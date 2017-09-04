#ifndef header_Mapping
#define header_Mapping

#include <iostream.h>
#include "Generic.h"
#include "Set.h"

class Mapping : public Set {
  
 public:
  // constructor & destructor ;
  Mapping(unsigned long sz);
  Mapping(Mapping& );
  ~Mapping();
  
 protected:
  // instance variables //
  const Generic ** values;
  
  // instance methods //
    // const unsigned long findIndex(const Generic& ) const;
  void grow(void);
  
 public:
  // accessing ;
  virtual const Generic& addMap(const Generic&, const Generic&);
  virtual const Generic & remove(const Generic& );
  // virtual const Boolean includes(const Generic& ) const;
  virtual const Generic& valueToKey(const Generic& ) const;
  
  // comparing ;
  //virtual const unsigned long hash() const;
  virtual const Boolean isEqualTo(const Generic& ) const;
  
  // printing ;
  virtual ostream& printOn(ostream& stream) const;
  
};
#endif
