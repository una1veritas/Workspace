#include <stream.h>
#include "test.h"

main() {
  SuperClass v, * p;

  p = new SubClass();

  cout << p->species() << "\n";

  //  cout << p->meme() << "\n";
}
