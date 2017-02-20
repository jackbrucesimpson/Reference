#include <iostream>

int main(){

  int ia[3];
  ia[0] = 5;
  ia[1] = 4;
  ia[2] = 3;

  // array elements can be accessed like the array was a pointer
  *ia = 6; // sets ia[0] = 6
  int *ip = ia;
  *ip = 2;
  ++ip;
  *ip = 22;
  *(++ip) = 4; // incrementing a pointer and using it at the same time

  std::cout << ia[0] << std::endl;
  std::cout << ia[1] << std::endl;
  std::cout << ia[2] << std::endl;

  int ia2[5] = {1,2,3,4,5};

  char s[] = {'s', 't', 'r', 'i', 'n', 'g', 0};
  char s2[] = "string";
  std::cout << s << std::endl;

  for (int i = 0; s[i]; ++i){
    std::cout << s[i] << std::endl;
  }

  // can also iterate through with a pointer
  for (char *cp = s; *cp; ++cp){
    std::cout << *cp << std::endl;
  }

  // range based loop - this will output the null value as its looking at the lengh of elements
  for(char c : s){
    if (c==0) break;
    std::cout << c << std::endl;
  }

  int v = 11;
  int w = 14;

  if (v > w) {
    std::cout << "greater" << std::endl;
  }
  else if (v > w) {
    std::cout << "less" << std::endl;
  }
  else {
    std::cout << "equal" << std::endl;
  }

  const int iONE = 1;
  const int iTWO = 2;
  const int iTHREE = 3;
  const int iFOUR = 4;

  int x = 0;

  switch(x) {
    case iONE:
      std::cout << "one" << std::endl;
      break;
      case iTWO:
        std::cout << "two" << std::endl;
        break;
      case iTHREE:
        std::cout << "three" << std::endl;
        break;
      case iFOUR:
        std::cout << "four" << std::endl;
        break;
    default:
      std::cout << "def" << std::endl;
      break;
  }

  int ii = 0;
  while (ii < 5) {
    std::cout << ii << std::endl;
    ++ii;
  }

  for (int i = 0; i < 5; ++i) {
    std::cout << i << std::endl;
  }

  int a[] = {1,2,3,4,5};

  for (int i : a) {
    std::cout << i << std::endl;
  }

  return 0;
}
