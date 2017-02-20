#include <iostream>
/* separate out declaration and implementation details
class Class1 {
private:
    int i;
public:
    int public_i = 12;
    void setvalue (const int value) {i = value;}
    int getvalue () const {return i;}
};
*/

class Class1 {
    int i;
public:
    int public_i = 55;
    void setvalue (const int value);
    int getvalue() const;
};

void Class1::setvalue(const int value) {
    i = value;
}

int Class1::getvalue() const {
    return i;
}

int main(){
    int i = 47;
    Class1 object1;
    object1.setvalue(i);
    std::cout << object1.getvalue() << std::endl;
    std::cout << object1.public_i << std::endl;
    object1.public_i = 5;
    std::cout << object1.public_i << std::endl;

  return 0;
}
