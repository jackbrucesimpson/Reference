#include <iostream>

int main(){
  int i = 11;

  int *ip;
  // address of i is pointed to ip
  ip = &i;
  int ii = 12;
  int *ip2 = &ii;

  // references
  int x = 7;
  int &y = x;

  std::cout << "Hello world" << std::endl;
  std::cout << ip << std::endl;
  std::cout << *ip << std::endl;
  std::cout << ip2 << std::endl;
  std::cout << *ip2 << std::endl;
  std::cout << y << std::endl;

  return 0;
}
