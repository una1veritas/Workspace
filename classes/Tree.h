#ifndef class_Tree
#define class_Tree

#include "Behavior.h"
#include "String.h"
#include "Bag.h"

class Tree : public Behavior {
 protected:
  String* label;
  Boolean isLeaf;
  Tree* left, * right;

  Tree(String& );
  Tree(String& , Tree& , Tree&);
  Tree(char * *); // ScanTree ;
  Tree(Set& , String& , Set& , String& );  // MakeDTree ;
  ~Tree();

 private:
  void findBestPattern(const Bag& , const Bag& , const int, String& ) const;
  void newpartition(String& , Bag&, Bag&, Bag&, Bag&, Bag&, Bag&) const;

 public:
  const unsigned long size(void) const;
  const String& decide(const String& ) const;
  // printing ;
  virtual ostream & printOn(ostream & stream) const;

};

#endif
