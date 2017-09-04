/*
  Bag.cc
  for class Bag
  */

#include "Bag.h"

// class initializarion ;
unsigned long Bag::max_print_size = 256;

// constructor & destructor;

 Bag::Bag(unsigned long sz) : Set(sz) {
   // initializing ``elements'' and ``element_count'' by Set(sz)

   occurrences = new unsigned long [basic_size];
   total = 0;
   // cout << " Bag with " <<sz << " elements.\n";
 }

 Bag::Bag(Bag& bag) {
   unsigned long i;
   
   basic_size = bag.basic_size;
   elements = new const Generic * [basic_size];
   occurrences = new unsigned long [basic_size];
   element_count = 0;
   total = 0;
   for (i = 0; i < bag.basic_size; i++) {
     if (bag.elements[i] == NULL) {
       elements[i] = NULL;
     } else {
       element_count++;
       elements[i] = bag.elements[i];
       occurrences[i] = bag.occurrences[i];
       total += bag.occurrences[i];
     }
   }
 }

 Bag::~Bag() {
   delete elements;
   delete occurrences;
 }

void Bag::grow(void) {
  const Generic** ptr1;
  unsigned long * ptr2;
  Bag* newbag;
  int i;

  newbag = new Bag(basic_size * 2 + 1);
  for (i = 0; i < basic_size; i++) {
    if (elements[i] != NULL)
      newbag->add(*(elements[i]),occurrences[i]);
  }
  ptr1 = elements;
  ptr2 = occurrences;
  elements = newbag->elements;
  occurrences = newbag->occurrences;
  newbag->elements = ptr1;
  newbag->occurrences = ptr2;
  basic_size = newbag->basic_size;
  // element_count = newset->element_count; // may be meaningless ;
  delete newbag;
}

const Generic & Bag::add(const Generic & anObj) {
  unsigned long i;

  // cerr << " adding " << anObj << " into " << i << "\n";
  total++;
  i = findIndex(anObj);
  if (elements[i] != NULL) {
    occurrences[i]++;
    return anObj;
  }
  elements[i] = & anObj;
  occurrences[i] = 1;
  element_count++;
  if (!(element_count < basic_size))
    grow();
  return anObj;
}

const Generic & Bag::add(const Generic & anObj, const unsigned long c) {
  unsigned long i;

  if (c == 0) 
    return anObj;
  i = findIndex(anObj);
  // cerr << " adding " << anObj << " into " << i << "\n";
  if (elements[i] == NULL) {
    elements[i] = & anObj;
    occurrences[i] = c;
    element_count++;
  } else {
    occurrences[i] += c;
  }
  total += c;
  if (!(element_count < basic_size))
    grow();
  return anObj;
}

const Generic & Bag::remove(const Generic & anObj) {
  unsigned long i;
  const Generic * theObj;
  i = findIndex(anObj);
  if ((theObj = elements[i]) == NULL)
    return (*theObj);
  occurrences[i]--;
  if (occurrences[i] == 0) {
    elements[i] = NULL;
    element_count--;
  }
  total--;
  return (*theObj);
}

const Boolean Bag::remove(const Generic & anObj, const unsigned long c) {
  unsigned long i;

  i = findIndex(anObj);
  if (elements[i] == NULL || occurrences[i] < c)
    return (Boolean) false;
  occurrences[i] -= c;
  if (occurrences[i] == 0) {
    elements[i] = NULL;
    element_count--;
  }
  total -= c;
  return (Boolean) true;
}

const unsigned long Bag::occurrencesOf(const Generic & anObj) const {
  unsigned long i;
  
  i = findIndex(anObj);
  if (elements[i] == NULL) {
    return (unsigned long) 0;
  } else {
    return (unsigned long) occurrences[i];
  }
}

const unsigned long Bag::size(void) const {
  return total;
}

const Boolean Bag::isEqualTo(const Generic& anObj) const {
  Bag* bagorg, * bagcopy;
  unsigned long i;

  bagorg = (Bag*) & anObj;
  if (element_count != bagorg->element_count)
    return (Boolean) false;
  bagcopy = new Bag(*bagorg);
  for (i = 0; i < basic_size; i++) {
    if (elements[i] != NULL) {
      bagcopy->remove(*(elements[i]), occurrences[i]);
      if (bagcopy->includes(*(elements[i]))) {
	cerr << "unmatched: " << (*bagcopy) << "\n";
	delete bagcopy;
	return (Boolean) false;
      }
    }
  }
  delete bagcopy;
  return (Boolean) true;
}

ostream & Bag::printOn(ostream & stream) const {
  unsigned long i, cnt;
  
  stream << "a Bag(";
  for (i = 0, cnt = 0; i < basic_size; i++) {
    if (cnt < max_print_size) {
      if (elements[i] != NULL) {
	stream << " ";
	stream << *(elements[i]);
	stream << "--";
	stream << occurrences[i];
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
