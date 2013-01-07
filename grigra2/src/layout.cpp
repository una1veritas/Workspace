#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <list>

#include "layout.h"

std::ostream &operator <<(std::ostream &out, Point &p) {
  p.print(out);
  return out;
}

class PointSet : public std::vector<Point> {
public:
  PointSet() : std::vector<Point>() {}
  ~PointSet() {}
  PointSet(const char *filename)
  : std::vector<Point>() {
    std::ifstream in(filename);
    if(in.is_open()) {
      int32 x, y;
      in >> x; in >> y;
      while(!in.eof()) {
        push_back(Point(x,y));
        in >> x; in >> y;
      }
    }
  }
  PointSet(PointSet &set) : std::vector<Point>(set.size()) {
    for(uint32 i=0; i < set.size(); ++i) {
      at(i) = set[i];
    }
  }
  PointSet(int32 size) : std::vector<Point>(size) {}
  void print(std::ostream &out) {
    for(iterator i = begin(); i != end(); i++) {
      i->print(out);
      out << std::endl;
    }
  }
  bool checkIndependent() {
    for(iterator i = begin(); i != end(); i++) {
      for(iterator j = begin(); j != end(); j++) {
        if(i == j) continue;
        if(*i == *j) {
          //std::cout << "NOT INDEPENDENT : " << *i << " " << *j << std::endl;
          return false;
        }
      }
    }
    return true;
  }
  PointSet &operator=(PointSet &set) {
    for(uint32 i=0; i<size(); ++i) {
      at(i) = set[i];
    }
    return *this;
  }
};
std::ostream &operator <<(std::ostream &out, PointSet &set) {
  set.print(out);
  return out;
}


class PointSequence : public std::vector<int32> {
  PointSet &set;
  class compare_x {
    PointSet &set;
  public:
    compare_x(PointSet &set) : set(set) {}
    bool operator ()(const int32 &a, const int32 &b) const {
      return set[a].getX() < set[b].getX();
    }
  };
  class compare_y {
    PointSet &set;
  public:
    compare_y(PointSet &set) : set(set) {}
    bool operator ()(const int32 &a, const int32 &b) {
      return set[a].getY() < set[b].getY();
    }
  };
public:
  ~PointSequence() {}
  PointSequence(PointSet &set, bool sortx)
  : std::vector<int32>(set.size()), set(set) {
    int32 j=0;
    for(iterator i=begin(); i!=end(); ++i) { *i = j++; }
    if(sortx) std::sort(begin(), end(), compare_x(set));
    else std::sort(begin(), end(), compare_y(set));
  }
  Point &ref(int32 i) { return set.at(at(i)); }
  void print(std::ostream &out) {
    for(iterator i=begin(); i!=end(); ++i)
      out << *i << " : " << set[*i] << std::endl;
  }
};
std::ostream &operator <<(std::ostream &out, PointSequence &seq) {
  seq.print(out);
  return out;
}

template <typename T>
class grid : public std::vector<T> {
  int32 grid_size;
public:
  grid() : std::vector<T>(), grid_size(0) {}
  grid(int32 size) : std::vector<T>(size*size), grid_size(size) {}
  T &ref(int32 x, int32 y) { return std::vector<T>::at(x + y * grid_size); }
  void print(std::ostream &out) {
    int32 j=0;
    for(typename std::vector<T>::iterator i=std::vector<T>::begin(); i!=std::vector<T>::end(); i++) {
      out << "  " << *i;
      if(++j % grid_size == 0) out << std::endl;
    }
  }
};
template <typename T>
std::ostream &operator <<(std::ostream &out, grid<T> &g) {
  g.print(out);
  return out;
}

class GridLayout {
  int32 grid_size;
  int32 checkPointPair(Point &p, Point &q) {
    if(p == q) return 1;
    if(p.getX() >= q.getX() && p.getY() <= q.getY()) return 2;
    return 0;
  }
  int32 prevX(int32 ix, int32 iy, PointSequence &x, PointSequence &y) {
    int32 t = y.ref(iy).getY();
    do { if(ix <= 0) return -1; } while(t < x.ref(--ix).getY());
    return ix;
  }
  int32 prevY(int32 ix, int32 iy, PointSequence &x, PointSequence &y) {
    int32 r = x.ref(ix).getX();
    do { if(iy <= 0) return -1; } while(r < y.ref(--iy).getX());
    return iy;
  }
  void listCountRects(PointSet &set, PointSequence &sortx, PointSequence &sorty) {
    grid<int32> cr(set.size());
    for(uint32 i=0; i<set.size(); ++i) {
      for(uint32 j=0; j<set.size(); ++j) {
        switch(checkPointPair(sortx.ref(i), sorty.ref(j))) {
          case 1: {
            int32 px = prevX(i, j, sortx, sorty);
            int32 py = prevY(i, j, sortx, sorty);
            int32 a = (px < 0 || py < 0) ? 0 : cr.ref(px, py);
            cr.ref(i,j) = a + 1;
          } break;
          case 2: {
            int32 px = prevX(i, j, sortx, sorty);
            int32 py = prevY(i, j, sortx, sorty);
            int32 a = (px < 0) ? 0 : cr.ref(px, j);
            int32 b = (py < 0) ? 0 : cr.ref(i, py);
            cr.ref(i,j) = (a < b ? a : b) + 1;
          } break;
        }
      }
    }
    //std::cout << "[COUNT]" << std::endl;
    //std::cout << cr << std::endl;
    calcTranslateTable(set, sortx, sorty, cr);
  }
  void calcTranslateTable(PointSet &set, PointSequence &x, PointSequence &y, grid<int32> &cr) {
    grid<Point> tt(set.size()), sd(set.size());
    for(uint32 i=0; i<set.size(); ++i) {
      for(uint32 j=0; j<set.size(); ++j) {
        if(cr.ref(i,j) == 0) continue;
        // CalcTranslate.
        //std::cout << "[CALC]" << i << " " << j << std::endl;
        switch(checkPointPair(x.ref(i), y.ref(j))) {
          case 1: {
            int32 px = prevX(i, j, x, y);
            int32 py = prevY(i, j, x, y);
            int32 r = (px < 0) ? 0 : x.ref(px).getX();
            int32 t = (py < 0) ? 0 : y.ref(py).getY();
            Point a = (px < 0 || py < 0) ? Point() : tt.ref(px, py);
            Point prep = (px < 0 || py < 0) ? Point() : ((sd.ref(px, py).getY()!=1) ? x.ref(px) : y.ref(py));
            Point da = x.ref(i).trans(prep, Point(r,t), grid_size);
            int32 len = da.length() + ((px < 0 || py < 0) ? 0 : sd.ref(px, py).getX());
            tt.ref(i, j) = da;
            sd.ref(i, j) = Point(len, 0);
          } break;
          case 2: {
            int32 px = prevX(i, j, x, y);
            int32 py = prevY(i, j, x, y);
            int32 r = (px < 0) ? 0 : x.ref(px).getX();
            int32 t = (py < 0) ? 0 : y.ref(py).getY();
            Point a = (px < 0) ? Point() : tt.ref(px, j);
            Point b = (py < 0) ? Point() : tt.ref(i, py);
            Point prea = (px < 0) ? Point() : (sd.ref(px, j).getY()!=1) ? x.ref(px) : y.ref(j);
            Point preb = (py < 0) ? Point() : (sd.ref(i, py).getY()!=1) ? x.ref(i) : y.ref(py);
            Point da = x.ref(i).trans(prea, Point(r,t), grid_size);
            Point db = y.ref(j).trans(preb, Point(r,t), grid_size);
            int32 xlen = da.length() + ((px < 0) ? 0 : sd.ref(px, j).getX());
            int32 ylen = db.length() + ((py < 0) ? 0 : sd.ref(i, py).getX());
            //std::cout << "LEN : " << xlen << " " << ylen << std::endl;
            if(xlen < ylen) {
              tt.ref(i, j) = da;
              sd.ref(i, j) = Point(xlen, 0);
            } else {
              tt.ref(i, j) = db;
              sd.ref(i, j) = Point(ylen, 1);
            }
          } break;
        }
      }
    }
    //std::cout << "[TRANS]" << std::endl;
    //std::cout << tt << std::endl;
    //std::cout << "[SUM TRANS]" << std::endl;
    //std::cout << sd << std::endl;
    applyLayout(set, x, y, cr, tt, sd);
  }
  void applyLayout(PointSet &set, PointSequence &x, PointSequence &y, grid<int32> &cr, grid<Point> &tt, grid<Point> &sd) {
    std::list<Point*> ap;
    int32 ix = set.size()-1, iy = set.size()-1, px = 0, py = 0;
    Point trans;
    PointSet tmp(set);
    //std::cout << "TABLE INFO : " << std::endl;
    //std::cout << cr;
    //std::cout << tt;
    //std::cout << sd;
    while(ix >= 0 && iy >= 0) {
      // Apply Layout
      trans = tt.ref(ix, iy);
      //std::cout << "[TO : " << ix << " " << iy << "]" << std::endl;
      if(cr.ref(ix, iy) == 0) {
        std::cout << "APPLY-LAYOUT BACKTRACE MISSING ROUTE!" << std::endl;
        std::cout << "STATUS : " << ix << " " << iy << std::endl;
        return;
      }
      //std::cout << "APPLY : " << ix << " " << iy << std::endl;
      switch(checkPointPair(x.ref(ix), y.ref(iy))) {
        case 1: {
          px = prevX(ix, iy, x, y); py = prevY(ix, iy, x, y);
          ap.push_front(&tmp[x[ix]]);
          ix = px; iy = py;
        } break;
        case 2: {
          px = prevX(ix, iy, x, y); py = prevY(ix, iy, x, y);
          if(sd.ref(ix,iy).getY()!=1) {
            //std::cout << "X POP" << std::endl;
            ap.push_front(&tmp[x[ix]]);
            ix = px;
          } else {
            //std::cout << "Y POP" << std::endl;
            ap.push_front(&tmp[y[iy]]);
            iy = py;
          }
        } break;
        default: {
          std::cout << "FAILED!" << std::endl;
          return;
        } break;
      }
      //std::cout << "[APPLY] : " << *(ap.front()) << " << " << trans << std::endl;
      //*(ap.front()) += trans;
      for(std::list<Point*>::iterator i=ap.begin(); i!=ap.end(); ++i) { *(*i) += trans; }
    }
    set = tmp;
  }
public:
  GridLayout(int32 grid_size) : grid_size(grid_size) {}
  ~GridLayout() {}
  PointSet &match(PointSet &p) {
    PointSequence sortx(p, true), sorty(p, false);
    //std::cout << "[SORT X]" << std::endl << sortx << std::endl;
    //std::cout << "[SORT Y]" << std::endl << sorty << std::endl;
    listCountRects(p, sortx, sorty);
    return p;
  }

  bool checkMatch(PointSet &p) {
    for(uint32 i=0; i<p.size(); ++i) {
      if(p[i].getX() % grid_size != 0 || p[i].getY() % grid_size != 0) {
        //std::cout << "NOT ON-GRID NO." << i << " : " << p[i] << std::endl;
        return false;
      }
    }
    return true;
  }
};

int32 main(int32 argc, char **argv) {
  if(argc != 3) {
    std::cout << "USAGE [layout.exe grid_size/int input-filename/string]" <<std::endl;
    return 0;
  }

  GridLayout gridlayout(atoi(argv[1]));
  PointSet pointset(argv[2]);

  // Display IN-POINT-SET
  //std::cout << "[IN]" << std::endl;
  //pointset.print(std::cout);
  //std::cout << std::endl;

  // Apply Matching
  if(!pointset.checkIndependent()) {
    std::cout << "POINT-SET IS NOT INDEPENDENT!" << std::endl;
  }
  if(gridlayout.checkMatch(pointset)) {
    std::cout << "POINT-SET IS ALREADY GRID-LAYOUTED!" << std::endl;
  }
  gridlayout.match(pointset);
  
  // Display OUT-POINT-SET
  //std::cout << "[INDEPENDENT?] " << pointset.checkIndependent() << std::endl;
  //std::cout << "[ONGRID?] " << gridlayout.checkMatch(pointset) << std::endl;
  // if(pointset.checkIndependent() && gridlayout.checkMatch(pointset)) {
  //  std::cout << "[OUT]" << std::endl;
  //  pointset.print(std::cout);
  //  std::cout << std::endl;
  //}

  std::cout << pointset;

  return 0;
}

