class SuperClass {
  char * str;
 public:
  virtual char * species() { return "SuperClass"; }
  SuperClass * self() { return this; }
  char * isit() { return "Am a SuperClass"; }
};

class SubClass : public SuperClass {
 public:
  char * species() { return "SubClass"; }
  char * isit() { return "Am a SubClass"; }
  char * meme() { return "Oops!"; }
  SubClass * self() { return this; }
};
