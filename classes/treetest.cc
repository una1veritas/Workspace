#include <stream.h>
#include "String.h"
#include "NodeTree.h"
#include "DTree.h"

main()
{
  DecisionTree * s, * t, * u;
  char d [100];

  s = new DecisionTree(new String("A"));
  t = new DecisionTree(new String("B"));
  u = new DecisionTree(new String("acc"),s,t);

  cout << *u << ";\n";
  cin >> d;
  cout <<  u->determine(String(d)) << "\n";

}
