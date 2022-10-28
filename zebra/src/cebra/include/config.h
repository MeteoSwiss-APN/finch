#ifndef CEBRA_CONFIG_H
#define CEBRA_CONFIG_H

/**
 * Sets the number of threads used in parallel regions.
*/
void set_threads(int n);

/**
 * Returns the number of threads used in parallel regions.
*/
int get_threads();

#endif