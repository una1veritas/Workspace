/*
  Mapping.cc
  for class Mapping
  */

#include "Mapping.h"

// constructor & destructor;

 Mapping::Mapping(unsigned long sz) : Set(sz) {
   // initializing ``elements'' and ``count'' by Set(sz) ;
   
   values  = new const Generic* [basic_size];
   // cout << " Mapping of size " <<sz << " elements.\n";
 }

 Mapping::Mapping(Mapping& map) : Set(map.basic_size) {
   unsigned long i;
   
   values = new const Generic * [basic_size];
   for (i = 0; i < map.basic_size; i++) {
     if (map.elements[i] == NULL) {
       elements[i] = NULL;
     } else {
       element_count++;
       elements[i] = map.elements[i];
       values[i] = map.values[i];
     }
   }
 }

 Mapping::~Mapping() {
   delete elements;
   delete values;
 }

void Mapping::grow(void) {
  const Generic ** ptr1, ** ptr2;
  Mapping * newmap;
  int i;

  newmap = new Mapping(basic_size * 2 + 1);
  for (i = 0; i < basic_size; i++) {
    if (elements[i] != NULL)
      newmap->addMap(*(elements[i]),*(values[i]));
  }
  ptr1 = elements;
  ptr2 = values;
  elements = newmap->elements;
  values = newmap->values;
  newmap->elements = ptr1;
  newmap->values = ptr2;
  basic_size = newmap->basic_size;
  // element_count = newmap->element_count; // may be meaningless ;
  delete newmap;
}

const Generic & Mapping::addMap(const Generic & anObj, const Generic & aVal) {
  unsigned long i;

  // cerr << " adding " << anObj << " into " << i << "\n";
  i = findIndex(anObj);
  if (elements[i] != NULL) 
    return anObj;
  elements[i] = & anObj;
  values[i] = & aVal;
  element_count++;
  if (!(element_count < basic_size))
    grow();
  return anObj;
}

const Generic & Mapping::remove(const Generic & anObj) {
  unsigned long i;
  const Generic * theObj;
  i = findIndex(anObj);
  if ((theObj = elements[i]) == NULL)
    return (*theObj);
  elements[i] = NULL;
  values[i] = NULL;
  element_count--;
  return (*theObj);
}


const Generic& Mapping::valueToKey(const Generic & anObj) const {
  unsigned long i;
  i = findIndex(anObj);
  if (elements[i] == NULL) 
    error("Mapping: Cannot find value to key,\n");
  else {
    return *values[i];
  }
}

const Boolean Mapping::isEqualTo(const Generic& anObj) const {
  Mapping * maporg, * mapcopy;
  unsigned long i;

  maporg = (Mapping *) & anObj;
  if (element_count != maporg->element_count)
    return (Boolean) false;
  mapcopy = new Mapping(*maporg);
  for (i = 0; i < basic_size; i++) {
    if (elements[i] != NULL) {
      mapcopy->remove(*(elements[i]));
      if ( mapcopy->includes(*(elements[i]))) {
	if ( mapcopy->valueToKey(*elements[i]) != (*values[i])) {
	  cerr << "unmatched: " << (*mapcopy) << "\n";
	  delete mapcopy;
	  return (Boolean) false;
	}
      }
    }
  }
  delete mapcopy;
  return (Boolean) true;
}

ostream & Mapping::printOn(ostream & stream) const {
  unsigned long i, cnt;
  stream << "a Mapping(";
  for (i = 0, cnt = 0; i < basic_size; i++) {
    if (cnt < MAX_PRINT_SIZE) {
      if (elements[i] != NULL) {
	stream << " ";
	stream << *(elements[i]);
	stream << "->";
	stream << *(values[i]);
	cnt ++;
      }
    } else {
      stream << " ...";
      break;
    }
  }
  stream << ") ";
  return stream;
}
