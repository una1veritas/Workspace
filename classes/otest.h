class Object {
 public:
  long size;
  
  Object() {
    size = 4;
  }

  void test() {
    cout << "Ok!\n";
  }
  virtual void print() {
    cout << "Object\n";
  }
};

class Integer : public Object {
 public:
  long value;

  Integer(int & i) {
    size = sizeof(value);
    value = i;
  }
  Integer(Integer & i) {
    size = sizeof(value);
    value = i.value;
  }
  
  Integer * operator=(Integer * i) {
    value = i->value;
    return this;
  }

  void print() {
    cout << value;
  }

};

