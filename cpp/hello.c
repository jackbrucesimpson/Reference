#include <stdio.h>
#include <stdlib.h>

int main() {
    int year = 2017;
    char letter = 'J';
    printf("Hello world, the year is %d and my name starts with %c\n", year, letter);
    printf("Type a letter\n");
    int c = getchar();
    printf("You typed: %c\n", c);
    return 0;
}
