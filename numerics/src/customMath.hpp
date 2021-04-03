#ifndef CUSTOMMATH_HPP
#define CUSTOMMATH_HPP

/*
Function for integer Powers stolen from:
https://stackoverflow.com/questions/1505675/power-of-an-integer-in-c
*/
inline int intPow(int x, int p) {
  if (p == 0)
    return 1;
  if (p == 1)
    return x;

  int tmp = intPow(x, p / 2);
  if (p % 2 == 0)
    return tmp * tmp;
  else
    return x * tmp * tmp;
}

#endif
