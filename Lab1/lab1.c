#include<stdio.h>
#include<sys/time.h>

int main(){
	struct timeval thisTime;

	gettimeofday(&thisTime, NULL);

	while(1){
		gettimeofday(&thisTime, NULL);
		double thisSecond = thisTime.tv_sec + ((double)thisTime.tv_usec / 1000000.0);


		printf("seconds: %lf\n",thisSecond);
	}
}