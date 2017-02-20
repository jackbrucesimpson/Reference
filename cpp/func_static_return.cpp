#include <iostream>
#include<string>

const std::string & func() {
  static std::string s = "Static string"; // store variable in static storage space
  return s;
}

void f() {
  std::cout << "This is a function" << std::endl;
}

int main(){
  std::cout << func() << std::endl;

  void (*fp)() = f; //function pointer
  fp();

  return 0;
}
