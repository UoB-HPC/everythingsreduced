// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctype.h>

static inline long long suffixed_atoll(const char *nptr) {
    char *mod              = strdup(nptr);
    const size_t  s        = strlen(mod);
    long long res2_power   = 0LL;
    long long res10_power  = 0LL;
    int ischar             = 1;
    int power_of_two_stage = 0;

    for(int p = s-1; p >= 0 && ischar; --p) {
        switch(tolower(mod[p])) {
        case 'b':
            power_of_two_stage = 1;
            break;
        case 'i':
            if(power_of_two_stage == 1) {
                power_of_two_stage = 2;
            }
            else {
                power_of_two_stage = 0;
            }
            break;
        case 'k':
            if(power_of_two_stage == 2) {
                res2_power += 10;
            }
            else {
                res10_power += 3;
            }
            power_of_two_stage = 0;
            break;
        case 'm':
            if(power_of_two_stage == 2) {
                res2_power += 20;
            }
            else {
                res10_power += 6;
            }
            power_of_two_stage = 0;
            break;
        case 'g':
            if(power_of_two_stage == 2) {
                res2_power += 30;
            }
            else {
                res10_power += 9;
            }
            power_of_two_stage = 0;
            break;
        default:
            ischar = 0;
        }
        mod[p+1] = 0;
    }
    long long power10 = 1;
    for(; res10_power > 0; --res10_power) {
        power10 *= 10LL;
    }
    const long long res = atof(mod) * ((1LL << res2_power) * power10);
    free(mod);
    return res;
}
