// Using a pthread mutex to get rid of the race condition
// Must be compiled with the  -fpermissive and -lpthread options
// g++ -fpermissive norace.cpp -lpthread
// Can ignore the two warnings

#include <iostream>
#include <pthread.h>
#include <unistd.h>

using namespace std;

struct  Info
{
	char position;
	int sleep;
	int carNum;
	int arrivalSleep;
};

static int carsWaiting = 0, northCrossed = 0, southCrossed = 0, maxNNBCars, northboundCars = 0, maxNSBCars, southboundCars = 0, maxNCars, carsNum = 0;
static pthread_mutex_t lock;
static pthread_mutex_t printLock;
static pthread_mutex_t northLock;
static pthread_mutex_t southLock;
static pthread_cond_t north = PTHREAD_COND_INITIALIZER;
static pthread_cond_t south = PTHREAD_COND_INITIALIZER;
static pthread_cond_t maxCars = PTHREAD_COND_INITIALIZER;

void *child_thread(void *arg) {
	Info* info;
	info = (Info*) arg;
	int wait = 0;

	sleep(info->arrivalSleep);

	pthread_mutex_lock(&printLock);
	cout << ((info->position == 'N') ?"Northbound":"Southbound")<< " car #"<< info->carNum<<" arrives at the tunnel."<<endl;
	pthread_mutex_unlock(&printLock);

	pthread_mutex_lock(&lock);
	while (carsNum >= maxNCars) {
		wait = 1;

		pthread_mutex_lock(&printLock);
		cout << "-- " << ((info->position == 'N') ? "Northbound" : "Southbound") << " car #" << info->carNum << " has to wait." << endl;
		pthread_mutex_unlock(&printLock);

		pthread_cond_wait(&maxCars, &lock);
	}
	pthread_mutex_unlock(&lock);

	if (info->position == 'N') {
		pthread_mutex_lock(&northLock);
		while (northboundCars >= maxNNBCars) {
			if (wait == 0) {
				pthread_mutex_lock(&printLock);
				cout << "-- Northbound car #" << info->carNum << " has to wait." << endl;
				pthread_mutex_unlock(&printLock);
			}
			wait = 1;
			pthread_cond_wait(&north, &northLock);
		}
		northboundCars++;
		pthread_mutex_unlock(&northLock);

		pthread_mutex_lock(&printLock);
		cout << "Northbound car #" << info->carNum << " enters the tunnel." << endl;
		pthread_mutex_unlock(&printLock);

		pthread_mutex_lock(&lock);
		carsNum++;
		pthread_mutex_unlock(&lock);

		sleep(info->sleep);

		pthread_mutex_lock(&northLock);
		northCrossed++;
		northboundCars--;

		pthread_mutex_lock(&printLock);
		cout <<"Northbound car #" << info->carNum << " exits the tunnel." << endl;
		pthread_mutex_unlock(&printLock);

		pthread_cond_signal(&north);
		pthread_mutex_unlock(&northLock);
	}
	else {
		pthread_mutex_lock(&southLock);
		while (southboundCars >= maxNSBCars) {
			if (wait == 0) {
				pthread_mutex_lock(&printLock);
				cout << "-- Southbound car #" << info->carNum << " has to wait." << endl;
				pthread_mutex_unlock(&printLock);
			}
			wait = 1;
			pthread_cond_wait(&south, &southLock);
		}
		southboundCars++;
		pthread_mutex_unlock(&southLock);

		pthread_mutex_lock(&printLock);
		cout << "Southbound car #" << info->carNum << " enters the tunnel." << endl;
		pthread_mutex_unlock(&printLock);

		pthread_mutex_lock(&lock);
		carsNum++;
		pthread_mutex_unlock(&lock);

		sleep(info->sleep);

		pthread_mutex_lock(&southLock);
		southCrossed++;
		southboundCars--;

		pthread_mutex_lock(&printLock);
		cout << "Southbound car #" << info->carNum << " exits the tunnel." << endl;
		pthread_mutex_unlock(&printLock);

		pthread_cond_signal(&south);
		pthread_mutex_unlock(&southLock);
	}

	pthread_mutex_lock(&lock);
	if (wait==1) {
		carsWaiting++;
	}

	carsNum--;
	pthread_cond_signal(&maxCars);
	pthread_mutex_unlock(&lock);

	return NULL;
}

int main() {
	pthread_t tid[100];
	Info info[100];
    pthread_mutex_init(&lock, NULL);
	pthread_mutex_init(&printLock, NULL);
	pthread_mutex_init(&northLock, NULL);
	pthread_mutex_init(&southLock, NULL);

	cin >> maxNCars;
	cin >> maxNNBCars;
	cin >> maxNSBCars;

	int numOfCars = 0, north = 1, south = 1, arrivalSleep;
	while (cin >> arrivalSleep >> info[numOfCars].position >> info[numOfCars].sleep) {
		if (numOfCars > 0) {
			info[numOfCars].arrivalSleep = info[numOfCars - 1].arrivalSleep + arrivalSleep;
		}
		else {
			info[numOfCars].arrivalSleep = arrivalSleep;
		}

		if (info[numOfCars].position=='N') {
			info[numOfCars].carNum = north++;
		}
		else {
			info[numOfCars].carNum = south++;
		}
		numOfCars++;
	}

	cout << "Maximum number of cars in the tunnel: " << maxNCars << endl;
	cout << "Maximum number of northbound cars: " << maxNNBCars << endl;
	cout << "Maximum number of southbound cars: " << maxNSBCars << endl;

	cout << "-----Start-----" << endl;
	for (int i = 0; i < numOfCars; i++) {

		pthread_create(&tid[i], NULL, child_thread, (void*)&info[i]);
	}
	for (int i = 0; i < numOfCars; i++) {
		pthread_join(tid[i], NULL);
	}

	cout << northCrossed << " northbound car(s) crossed the tunnel." << endl;
	cout << southCrossed << " southbound car(s) crossed the tunnel." << endl;
	cout << carsWaiting << " car(s) had to wait." << endl;

	return 0;
}
