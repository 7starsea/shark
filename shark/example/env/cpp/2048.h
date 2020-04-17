#ifndef SHARE_EXAMPLE_ENV_2048_H
#define SHARE_EXAMPLE_ENV_2048_H

#include <vector>
#include "inc/cnarray.hpp"
#include "crandom.h"


typedef cndarray<int, 2> int_2darray;
typedef cndarray<uint8_t, 2> uint8_2darray;
typedef cndarray<uint8_t, 3> uint8_3darray;

/// the game 2048 implementation algorithm is copied from https://github.com/nneonneo/2048-ai

namespace Game2048N{

typedef uint64_t board_t;
typedef uint16_t row_t;

static const board_t ROW_MASK = 0xFFFFULL;
static const board_t COL_MASK = 0x000F000F000F000FULL;


void to_m_board(board_t board, uint8_2darray & arr);
void to_m_board(board_t board, uint8_3darray & arr);
board_t to_c_board(const uint8_2darray & arr);

// Transpose rows/columns in a board:
//   0123       048c
//   4567  -->  159d
//   89ab       26ae
//   cdef       37bf
board_t transpose(board_t x);

static inline board_t unpack_col(row_t row) {
    board_t tmp = row;
    return (tmp | (tmp << 12ULL) | (tmp << 24ULL) | (tmp << 36ULL)) & COL_MASK;
}

static inline row_t reverse_row(row_t row) {
    return (row >> 12) | ((row >> 4) & 0x00F0)  | ((row << 4) & 0x0F00) | (row << 12);
}


class Game2048{
public:
	Game2048();

	void seed(int seed_id){ rand_.seed(seed_id); rand24_.seed(seed_id + 1000);}
	void reset();
	float step(int action);

	std::vector<int> legal_actions()const;
	float score()const{return score_; }
	int max_value()const{ return 1 << get_max_rank(board_); }

	static int get_max_rank(board_t board);
	static int count_distinct_tiles(board_t board);

	// Count the number of empty positions (= zero nibbles) in a board.
	// Precondition: the board cannot be fully empty.
	static int count_empty(board_t x);

	board_t insert_tile_rand(board_t board, board_t tile);

protected:
	board_t board_;
	float score_;
	CRandom rand_;
	CRandom rand24_;

protected:
	board_t draw_tile(){return rand24_.sample() < 90 ? 1 : 2; }
protected:
	static board_t execute_move(int move, board_t board);	
	static board_t execute_move_0(board_t);
	static board_t execute_move_1(board_t);
	static board_t execute_move_2(board_t);
	static board_t execute_move_3(board_t);


};


}
#endif


