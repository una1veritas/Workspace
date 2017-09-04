#ifndef header_Integer
#define header_Integer

class Integer : public Generic {

 protected:
  long value;

 public:
  // constructor & destructor ;
  Integer(const long val = 0) {
    value = val;
  }
  Integer(const Integer & anInt) {
    value = anInt.value;
  }

  // instance methods;
  const Boolean isEqualTo(const Generic & s) const {
    return (Boolean) (value == ((Integer *)(& s))->value);
  }

  virtual const unsigned long hash() const { 
    return (unsigned long) value; 
  }

  Integer & operator = (const long val) {
    value = val;
    return *this;
  }

  Integer & operator = (const Integer & anInt) {
    value = anInt.value;
    return *this;
  }

  Integer & operator + (const Integer & anInt) {
    return *(new Integer(value + anInt.value));
  }
  

  // printing ;
  ostream & printOn(ostream & stream) const {
    stream << value;
    return stream;
  }

};

#endif
