#include <stdio.h>
#include <stdlib.h>

// ... define function 'add' ...

int main(int argc, char **argv) {
  int a, b, c;        // We've chosen static allocation here for host storage..
  int *da, *db, *dc;  // ...but device storage must be dynamically allocated
  a = atoi(argv[1]);  // Read the addends from the command line args
  b = atoi(argv[2]);

  // ... manage memory ...

  // ... move data ...

  add<<<1,1>>>(da, db, dc);

  // ... move data ...

  printf("%d + %d -> %d\n", a, b, c);

  // ... manage memory ...
}
