#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define VERSION 3

int main() {
    char password[] = "password";
    int len = strlen(password);

    printf("Password is: '%s'\n", password);
    printf("Num letters: %d\n", len);
    printf("%c\n", password[0]);

    char first[] = "I would like to go ";
    char second[] = "from here to here\n";
    char str_lens = strlen(first) + strlen(second);
    char storage[str_lens];
    strcpy(storage, first);
    strcat(storage, second);

    printf("Version number %d\n", VERSION);

    int password_match = strcmp(password, "password");
    int password_mistake = strcmp(password, "wrong");

    printf("Password comparison: %d %d\n", password_match, password_mistake);

    printf("%s\n", storage);

    float r = sqrt(2.0);
    float p = pow(2.0, 8.0);

    srand(66);
    //srand((unsigned)time(NULL));
    int random = rand();

    printf("%f %f %d\n", r, p, random);

    return 0;
}
