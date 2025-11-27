#include <stdio.h>

int main()
{
	int x = 5;
	int *p = &x;
	*p = 6;
	int** q = &p;
	int*** r = &q;

	printf("%d\n", *p); // 6
	printf("%d\n", q); // 6291064
	printf("%d\n", p); // 6291076
	printf("%d\n", r); // 6291056
}