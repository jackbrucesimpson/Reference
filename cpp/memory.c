#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024

int main() {
    char *sto;
    sto = (char *)malloc(sizeof(char)*SIZE);

    if (sto == NULL) {
        puts("Memory error");
        return 1;
    }

    return 0;
}
