#ifndef LAYOUT_HPP
#define LAYOUT_HPP

#include <iostream>

namespace gridlayout {

  template<typename TYPE>
  class PointSet;

  template<typename TYPE>
  class PointSetProxy;

  //
  // Grid Layout Function
  // IN - pointset : input-pointset.
  // IN - gridsize : gridsize.
  //
  // OUT - pointset : output-pointset.
  // OUT - error-value : 0 on success, -1 on failed.
  //
  template<typename TYPE>
  int GridLayout(PointSetProxy<TYPE> *pointset, const TYPE gridsize);
  
  //
  // [Tamplate]
  // TYPE : type of a point-element.
  // 
  // [Class] Point Set
  //
  // <Constructor>
  // std::istream& : point-set-text-stream
  // const TYPE*, const size_t : pointset-array and length-of-array
  // ex. { x0, y0, x1, y1, x2, y2, ... , xn, yn } , length = 2*n
  //
  template<typename TYPE>
  class PointSetProxy {
    friend int GridLayout<TYPE>(PointSetProxy<TYPE> *pointset, const TYPE gridsize);
    PointSet<TYPE> *set_p;
  public:
    PointSetProxy();
    PointSetProxy(const char *filename);
    PointSetProxy(std::istream &sIn);
    PointSetProxy(const TYPE *pset, const size_t length);
    ~PointSetProxy();

    size_t length();

    void print();
    void print(std::ostream &sOut);
    void print(TYPE *out_p, const size_t maxlen);
  };
}

#endif
