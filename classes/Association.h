#ifndef header_Association
#define header_Association

class Association : public Generic {

 public:
  long key, value;

 public:
  Association(long from, long to) {
    key = from;
    value = to;
  }
  
  virtual const unsigned long hash() const { 
    return (unsigned long) key*0x10000 + value;
  }
  
  virtual const Boolean isEqualTo(const Generic & obj) const { 
    return (Boolean) ((key == ((Association *) &obj)->key) 
      && (value == ((Association *) & obj)->value));
  }

  // printing;
  ostream& printOn(ostream& stream) const {
    stream << "(" << key << " -> " << value << ")";
    return stream; 
  }
  
};

#endif
