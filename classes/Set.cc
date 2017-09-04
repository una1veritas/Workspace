#include "Set.h"

// constructor & destructor;

 Set::Set(const unsigned long sz) {
   unsigned long i;
   
   basic_size = ((sz < 4)? 4 : sz);
   elements = new const Generic * [basic_size];
   element_count = 0;
   for (i = 0; i < sz; i++) {
     elements[i] = NULL;
   }
   // cout << "Now, Set " << sz << "\n";
 }

 Set::Set(const Set& set) {
   unsigned long i;

   basic_size = set.basic_size;
   elements = new const Generic * [basic_size];
   element_count = 0;
   for (i = 0; i < set.basic_size; i++) {
     if (set.elements[i] == NULL) {
       elements[i] = NULL;
     } else {
       element_count++;
       elements[i] = set.elements[i];
     }
   }
 }

 Set::~Set() {
   // cerr << "deleting Set \n";
   delete [] elements;
 }


const unsigned long Set::findIndex(const Generic & anObj) const {
  // return the index of anObj if found, otherwise of NULL ;

  unsigned long index, cnt;

  index = anObj.hash();
  for (cnt = 0; cnt < basic_size; cnt++, index++) {
    index = index % basic_size;
    if (elements[index] == NULL) {
      return index;
    } else {
      if ((*(elements[index])) == anObj) {
	return index;
      }
    }
  }
  // any Set must have at least one NULL in elements[];
  error(" Error: Set findIndex() :no NULL in elements[]\n");
  return 0;
}

void Set::grow(void) {
  const Generic** objptr;
  Set* newset;
  int i;

  //cerr << " growing... ";
  newset = new Set(basic_size * 2 + 1);
  for (i = 0; i < basic_size; i++) {
    if (elements[i] != NULL)
      newset->add(*(elements[i]));
  }
  objptr = elements;
  elements = newset->elements;
  newset->elements = objptr;
  basic_size = newset->basic_size;
  // element_count = newset->element_count; // may be meaningless ;
  delete newset;
}

const Generic & Set::add(const Generic & anObj) {
  unsigned long i;

  i = findIndex(anObj);
  if (elements[i] != NULL) {
    // already included. ;
    return anObj;
  }
  elements[i] = (Generic *) &anObj;

  element_count++;
  if (!(element_count < basic_size))
    grow();
  return anObj;
}

void Set::addAll(const Set & set) {
  unsigned long i;
  for (i = 0; i < set.basic_size; i++) {
    if (set.elements[i] != NULL)
      add(*(set.elements[i]));
  }
}


const Boolean Set::removeAll(void) {
  unsigned long i;
  for (i = 0; (element_count > 0) && (i < basic_size); i++) 
    if (elements[i] != NULL) {
      elements[i] = NULL;
      element_count--;
    }
  return (Boolean) true;
}

const Boolean Set::includes(const Generic & anObj) const {
  return (Boolean) (elements[findIndex(anObj)] != NULL);
}

const unsigned long Set::size(void) const {
  return element_count;
}


const Boolean Set::isEqualTo(const Generic& s) const {
  Set* set;
  unsigned long i;

  set = (Set* ) &s;
  if ( element_count != set->element_count ) 
    return (Boolean) false;

  for ( i = 0; i < basic_size; i++) {
    if (elements[i] != NULL) {
      if (! set->includes(*(elements[i])))
	return false;
    }
  }
  return true;
}


