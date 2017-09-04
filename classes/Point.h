#ifndef header_Point
#define header_Point
#include <math.h>

#include <iostream.h>
#include "Generic.h"

class Point : public Generic {
 public:
  int x, y;

  Point() {
    x = 0;
    y = 0;
  }

  Point(int px, int py) {
    x = px;
    y = py;
  }

 public:
  const float distanceTo(const Point& p) const {
    return sqrt( ((p.x - x)*(p.x - x)) + ((p.y - y)*(p.y - y)) );
  }

  const bool isEqualTo(const Generic & obj) const { 
    Point * p;
    p = (Point* ) &obj;
    //cerr << " (Generic::isEqualTo) ";
    return (bool) (x == p->x && y == p->y);
  }

  const unsigned long hash() const {
    return x^y;
  }

  const bool isLeftThan(const Point& p) const {
    return x < p.x;
  }

  const bool isBelowThan(const Point& p) const {
    return y < p.y;
  }

  const bool isInRect(const Point & p1, const Point & p2) const {
    return (p1.x <= x && x <= p2.x) &&
      (p1.y <= y && y <= p2.y);
  }
  
  friend const bool operator< (const Point& p1, const Point& p2) { 
    return (p1.x < p2.x) || ( (!(p1.x > p2.x)) && p1.y < p2.y);
  }

  Point & operator = (const Point & p) {
    x = p.x;
    y = p.y;
    return *this;
  }

  // printing;
  ostream& printOn(ostream& stream) const {
    stream << " " <<  x << ", " << y ;
    return stream; 
  }

};

#endif
