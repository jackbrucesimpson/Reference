#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void minus10 (int *v);
char *longer (char *s1, char *s2);

int main() {

    int a = 65;
    int s = sizeof(a); // size in bytes;
    printf("Variable a is: %d and size is %d and memory location is %p\n", a, s, &a);

    int pokey = 11;
    int *p;
    p = &pokey;
    printf("Address of pokey is %p and value is %d\n", p, *p);

    char letter = 'L';
    char *pl;
    pl = &letter;
    *pl = 'J';
    printf("Changed letter to %c\n", letter);

    int value = 20;
    minus10(&value);
    printf("Value is: %d\n", value);

    char *result;
    result = longer("first", "second");
    printf("The longer string text is %s\n", result);

    return 0;
}

void minus10 (int *v) {
    *v = *v - 10;
}

char *longer(char *s1, char *s2) {
    int len1, len2;
    len1 = strlen(s1);
    len2 = strlen(s2);

    if (len1 > len2) {
        return s1;
    }
    else {
        return s2;
    }
}
