#ifndef SHARE_EXAMPLE_ENV_CRANDOM_H
#define SHARE_EXAMPLE_ENV_CRANDOM_H

#include <random>

class CRandom{
public:
	CRandom(int low, int high, int seed): rng_(seed), u_dist_(low, high) {} 

	inline void seed(int seed_id){
		rng_.seed(seed_id);
    	u_dist_.reset();
	}

	inline int sample(int mod=0){ return mod > 0 ? (u_dist_(rng_) % mod) : u_dist_(rng_); }

private:
	std::mt19937 rng_;
    std::uniform_int_distribution<int> u_dist_;
};


#endif