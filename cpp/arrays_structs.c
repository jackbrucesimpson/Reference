#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ROWS 4
#define COLUMNS 4

int main() {

    int nums[] = {1,2,3,4,5};
    printf("My number is %d\n", nums[0]);

    char name[] = "Jack";
    int n = 0;
    while (name[n] != '\0') {
        putchar(name[n]);
        n++;
    }
    putchar('\n');

    // can shorten
    n = 0;
    while (name[n]) {
        putchar(name[n]);
        n++;
    }
    putchar('\n');

    int grid[ROWS][COLUMNS];
    int x, y;

    for (x=0; x<ROWS;x++) {
        for (y=0; y<COLUMNS;y++) {
            grid[x][y] = y;
        }
    }

    for (x=0; x<ROWS;x++) {
        for (y=0; y<COLUMNS;y++) {
            printf("%d.%d: %d\t", x, y, grid[x][y]);
        }
        putchar('\n');
    }

    struct record {
        int account;
        float balance;
    };

    struct record mybank;
    mybank.account = 11;
    mybank.balance = 22.3;

    printf("My Bank account: %d and balance %f\n", mybank.account, mybank.balance);

    struct person {
        char name[32];
        int age;
    };

    struct person president;
    strcpy(president.name, "Washington");
    president.age = 67;

    printf("President %s, aged %d\n", president.name, president.age);

    return 0;
}
