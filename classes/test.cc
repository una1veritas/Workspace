#include <iostream.h>
#include "Generic.h"
#include "String.h"
#include "Set.h"
#include "Bag.h"
#include "Mapping.h"
#include "Graph.h"

main() {
  Generic* obj, * another;
  String s("Meee!"), t("Key!"), d("200");
  Mapping map(10);
  Mapping * map2;

  obj = new String("The String!");
  another = new Generic;

  cout << map << "\n" << "Size " << map.size() << "\n";
  map.addMap(s, *obj);
  cout << map << "\n" << "Size " << map.size() << "\n";
  map.addMap(s, *obj);
  cout << map << "\n" << "Size " << map.size() << "\n";
  map.addMap(t, d);
  cout << map << "\n" << "Size " << map.size() << "\n";
  map.addMap(*another, d);
  cout << map << "\n" << "Size " << map.size() << "\n";
  map.addMap(*another, t);
  cout << map << "\n" << "Size " << map.size() << "\n\n";

  map2 = new Mapping(map);
  map2->remove(String("Boy!!"));
  cout << * map2 << "\n";
  cout << "equality: " << (map == (*map2)) << "\n";

  map2->remove(String("Meee!"));
  cout << *map2 << "\n";
  map.remove(String("Meee!"));
  cout << "equality: " << (map == (*map2)) << "\n";

}
