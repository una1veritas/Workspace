/*
 * layout.h
 *
 *  Created on: 2013/01/07
 *      Author: sin
 */

#ifndef LAYOUT_H_
#define LAYOUT_H_

#include <stdint.h>

#define int8 		int8_t
#define uint8 		uint8_t
#define byte 		uint8_t
#define int16 		int16_t
#define uint16 		uint16_t
#define word 		uint16_t
#define int32 		int32_t
#define uint32 		uint32_t
#define int64 		int64_t

class Point {
  int32 x, y;
public:
  Point() : x(0), y(0) {}
  Point(int32 x, int32 y) : x(x), y(y) {}
  Point(const Point &p) : x(p.x), y(p.y) {}
  ~Point() {}
  int32 getX() const { return x; }
  int32 getY() const { return y; }
  void setX(int32 x) { this->x = x; }
  void setY(int32 y) { this->y = y; }
  Point trans(Point q, Point rt, int32 grid) {
    Point t(*this - q), rt2(rt - q), r;
    if(t.x % grid == 0) {
      if(t.x == 0) { r.x = grid; } else { r.x = 0; }
    } else {
      int32 u = (abs(t.x) % grid) * (t.x > 0 ? -1 : 1);
      int32 v = (grid - abs(u)) * (t.x > 0 ? 1 : -1);
      if(t.x + u <= rt2.x) { r.x = v; }
      else if(t.x + v <= rt2.x) { r.x = u;}
      else { r.x = abs(u) < abs(v) ? u : v; }
    }
    if(t.y % grid == 0) {
      if(t.y == 0) { r.y = grid; } else { r.y = 0; }
    } else {
      int32 u = (abs(t.y) % grid) * (t.y > 0 ? -1 : 1);
      int32 v = (grid - abs(u)) * (t.y > 0 ? 1 : -1);
      if(t.y + u <= rt2.y) { r.y = v; }
      else if(t.y + v <= rt2.y) { r.y = u; }
      else { r.y = abs(u) < abs(v) ? u : v; }
    }
    //std::cout << "[IN] : " << q.x << " " << q.y << " / " << rt.x << " " << rt.y << std::endl;
    //std::cout << "[TRANS] : " << x << " " << y << " / " << r.x << " " << r.y << std::endl;
    return r;
  }
  int32 sum() const { return x+y; }
  int32 length() const { return abs(x)+abs(y); }
  bool operator ==(Point &p) { return (x == p.x) && (y == p.y); }
  Point &operator =(Point p) { x=p.x; y=p.y; return *this; }
  Point operator +(Point &p) { Point q(x + p.x, y + p.y); return q; }
  Point &operator +=(Point p) { x += p.x; y += p.y; return *this; }
  Point operator -(Point &p) { Point q(x - p.x, y - p.y); return q; }
  void print(std::ostream &out) { out << x << " " << y; }
};


#endif /* LAYOUT_H_ */
