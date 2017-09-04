#ifndef header_Generic
#define header_Generic

#include <iostream.h>
#include <stdlib.h>

class Generic {
  // class methods ;
 private:
  static const void* class_identity() { 
    return &class_identity; 
  }
  
  // instance variables //
 protected:
  unsigned long basic_size;
  
  // instance methods //
 public:
  // comparing //
  virtual const void* classKind() const { 
    return class_identity(); 
  }

  virtual const unsigned long hash() const { 
    return (unsigned long) this; 
  }
  
  virtual const bool isEqualTo(const Generic & obj) const { 
    //cerr << " (Generic::isEqualTo) ";
    return (bool) (this == (& obj));
  }
  
  friend const bool
    operator == (const Generic & obj1, const Generic & obj2) { 
      if (obj1.classKind() == obj2.classKind()) 
	return obj1.isEqualTo(obj2);
      else {
	//cerr << "non same-kind objs ";
	return false;
      }
    }
  
  // error // 
  void error(char * s) const { 
    cerr << s;
    exit(1);
  }

  friend const bool
    operator != (const Generic & obj1, const Generic & obj2) {
      return (bool) !(obj1 == obj2);  
    }
  
  // printing //
  virtual ostream & printOn(ostream & stream) const {
    return stream << " Generic()";
  }
  
  // input & output ;
  friend ostream & operator << (ostream & stream, const Generic & obj) {
    return obj.printOn(stream);
  }
  
};

#endif
