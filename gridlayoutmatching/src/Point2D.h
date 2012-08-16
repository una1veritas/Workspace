/*
 * Node.h
 *
 *  Created on: 2012/04/26
 *      Author: sin
 */

#ifndef POINT2D_H_
#define POINT2D_H_

#include <math.h>
#include <iostream>

using namespace std;

class Point2D {
public:
	long x, y;

public:
	Point2D(const Point2D & p) :
			x(p.x), y(p.y) {
	}

	Point2D(const int px, const int py) :
			x(px), y(py) {
	}

	Point2D(void) {
		x = 0;
		y = 0;
	}

	Point2D operator+(const Point2D & p) {
		return Point2D(x+p.x, y+p.y);
	}

	Point2D operator-(const Point2D & p) {
		return Point2D(x-p.x, y-p.y);
	}

	const bool operator<(const Point2D & v) const {
		return (x < v.x) | ((x == v.x) && (y < v.y));
	}

	const bool operator==(const Point2D & v) const {
		return (x == v.x) && (y == v.y);
	}

	const bool operator!=(const Point2D & v) const {
		return (x != v.x) || (y != v.y);
	}


	const bool inRect(const Point2D & bl, const Point2D & tr) const {
		return (x >= bl.x && x <= tr.x) && (y >= bl.y && y <= tr.y);
	}

	const unsigned long norm1() {
		return abs(x)+abs(y);
	}

	// printing;
	ostream& printOn(ostream& stream) const {
		stream << "(" << x << ", " << y << ")";
		return stream;
	}

	// input & output ;
	friend ostream & operator <<(ostream & stream, const Point2D & obj) {
		return obj.printOn(stream);
	}
};
/*
class Rectangle {
public:
	Point2D bottomLeft, topRight;
public:
	Rectangle(Point2D & bl, Point2D & tr) : bottomLeft(bl), topRight(tr) {}
};
*/
#endif /* POINT2D_H_ */
