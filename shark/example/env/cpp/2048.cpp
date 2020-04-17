#include "2048.h"

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <iostream>

namespace Game2048N{

// Heuristic scoring settings
static const float SCORE_LOST_PENALTY = 200000.0f;
static const float SCORE_MONOTONICITY_POWER = 4.0f;
static const float SCORE_MONOTONICITY_WEIGHT = 47.0f;
static const float SCORE_SUM_POWER = 3.5f;
static const float SCORE_SUM_WEIGHT = 11.0f;
static const float SCORE_MERGES_WEIGHT = 700.0f;
static const float SCORE_EMPTY_WEIGHT = 270.0f;

/* We can perform state lookups one row at a time by using arrays with 65536 entries. */

/* Move tables. Each row or compressed column is mapped to (oldrow^newrow) assuming row/col 0.
 *
 * Thus, the value is 0 if there is no move, and otherwise equals a value that can easily be
 * xor'ed into the current board state to update the board. */
static	row_t row_left_table [65536];
static	row_t row_right_table[65536];
static	board_t col_up_table[65536];
static	board_t col_down_table[65536];
static	float heur_score_table[65536];
static	float score_table[65536];


static bool s_is_init = false;

// score a single board heuristically
static float score_heur_board(board_t board);
// score a single board actually (adding in the score from spawned 4 tiles)
static float score_board(board_t board);

static float score_helper(board_t board, const float* table);

static void init_tables();


	uint8_t log2int(int x){
    	uint8_t ret = 0;
	    while(x > 0){
	        ret += 1;
	        x >>= 1;
	    }
	    return ret - 1;
	}

	void to_m_board(board_t board, uint8_2darray & arr) {
	    for(int i=0; i<4; ++i) {
	        for(int j=0; j<4; ++j) {
	            uint8_t powerVal = (board) & 0xf;
	            arr.ix(i, j) = powerVal; ///(powerVal == 0) ? 0 : 1 << powerVal; 
	            board >>= 4;
	        }
	    }
	}

	void to_m_board(board_t board, uint8_3darray & arr) {
	    for(int i=0; i<4; ++i) {
	        for(int j=0; j<4; ++j) {
	            uint8_t powerVal = (board) & 0xf;
	            arr.ix((int)powerVal, i, j) = 1; ///(powerVal == 0) ? 0 : 1 << powerVal; 
	            board >>= 4;
	        }
	    }
	}


	board_t to_c_board(const uint8_2darray & arr){
		board_t board = 0;
		const int rows = arr.shape(0), cols = arr.shape(1);
		int ind = 0;
		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				board |= arr.ix(i, j) << (4 * ind);
				++ind;
			}
		}
		return board;
	}

	board_t transpose(board_t x){
	    board_t a1 = x & 0xF0F00F0FF0F00F0FULL;
	    board_t a2 = x & 0x0000F0F00000F0F0ULL;
	    board_t a3 = x & 0x0F0F00000F0F0000ULL;
	    board_t a = a1 | (a2 << 12) | (a3 >> 12);
	    board_t b1 = a & 0xFF00FF0000FF00FFULL;
	    board_t b2 = a & 0x00FF00FF00000000ULL;
	    board_t b3 = a & 0x00000000FF00FF00ULL;
	    return b1 | (b2 >> 24) | (b3 << 24);
	}


	float score_helper(board_t board, const float* table) {
	    return table[(board >>  0) & ROW_MASK] +
	           table[(board >> 16) & ROW_MASK] +
	           table[(board >> 32) & ROW_MASK] +
	           table[(board >> 48) & ROW_MASK];
	}

	float score_heur_board(board_t board) {
	    return score_helper(          board , heur_score_table) +
	           score_helper(transpose(board), heur_score_table);
	}

	float score_board(board_t board) {
	    return score_helper(board, score_table);
	}

	Game2048::Game2048()
	: board_(0)
	, score_(0)
	, rand_(0, 10000000, time(NULL))
	, rand24_(0, 100, time(NULL) + 1000)
	{
		if(!s_is_init){
			init_tables();
			s_is_init = true;
		}
	}

	void Game2048::reset(){
		score_ = 0;

		board_t board = draw_tile() << (4 * rand_.sample(16));
		board_ = insert_tile_rand(board, draw_tile());
	}
	float Game2048::step(int action){
		//// 0,1,2,3
		board_t board = execute_move(action, board_);
		///float reward = -4;
        ///if(board != board_){
            const float new_score = score_board(board);
            /// const bool done = legal_actions().size() 0;
            float reward = new_score - score_;
            score_ = new_score;

            board_ = insert_tile_rand(board, draw_tile());
        ///}
		return reward;
	}

	std::vector<int> Game2048::legal_actions()const{
		std::vector<int> x;
		if(get_max_rank(board_) < 16){   /// 1 << 16 = 65536
	        for(int move = 0; move < 4; ++move) {
	            if(execute_move(move, board_) != board_){
	            	x.push_back(move);
	            }
	        }
    	}
        return x;
	}

	board_t Game2048::insert_tile_rand(board_t board, board_t tile) {
	    int index = rand_.sample(count_empty(board));
	    board_t tmp = board;
	    while (true) {
	        while ((tmp & 0xf) != 0) {
	            tmp >>= 4;
	            tile <<= 4;
	        }
	        if (index == 0) break;
	        --index;
	        tmp >>= 4;
	        tile <<= 4;
	    }
	    return board | tile;
	}

	void init_tables(){

		for (unsigned row = 0; row < 65536; ++row) {
		        unsigned line[4] = {
		                (row >>  0) & 0xf,
		                (row >>  4) & 0xf,
		                (row >>  8) & 0xf,
		                (row >> 12) & 0xf
		        };

		        // Score
		        float score = 0.0f;
		        for (int i = 0; i < 4; ++i) {
		            int rank = line[i];
		            if (rank >= 2) {
		                // the score is the total sum of the tile and all intermediate merged tiles
		                score += (rank - 1) * (1 << rank);
		            }
		        }
		        score_table[row] = score;


		        // Heuristic score
		        float sum = 0;
		        int empty = 0;
		        int merges = 0;

		        int prev = 0;
		        int counter = 0;
		        for (int i = 0; i < 4; ++i) {
		            int rank = line[i];
		            sum += std::pow( (float)(rank), SCORE_SUM_POWER);
		            if (rank == 0) {
		                empty++;
		            } else {
		                if (prev == rank) {
		                    ++counter;
		                } else if (counter > 0) {
		                    merges += 1 + counter;
		                    counter = 0;
		                }
		                prev = rank;
		            }
		        }
		        if (counter > 0) {
		            merges += 1 + counter;
		        }

		        float monotonicity_left = 0;
		        float monotonicity_right = 0;
		        for (int i = 1; i < 4; ++i) {
		            if (line[i-1] > line[i]) {
		                monotonicity_left += pow(line[i-1], SCORE_MONOTONICITY_POWER) - pow(line[i], SCORE_MONOTONICITY_POWER);
		            } else {
		                monotonicity_right += pow(line[i], SCORE_MONOTONICITY_POWER) - pow(line[i-1], SCORE_MONOTONICITY_POWER);
		            }
		        }

		        heur_score_table[row] = SCORE_LOST_PENALTY +
		            SCORE_EMPTY_WEIGHT * empty +
		            SCORE_MERGES_WEIGHT * merges -
		            SCORE_MONOTONICITY_WEIGHT * std::min(monotonicity_left, monotonicity_right) -
		            SCORE_SUM_WEIGHT * sum;

		        // execute a move to the left
		        for (int i = 0; i < 3; ++i) {
		            int j;
		            for (j = i + 1; j < 4; ++j) {
		                if (line[j] != 0) break;
		            }
		            if (j == 4) break; // no more tiles to the right

		            if (line[i] == 0) {
		                line[i] = line[j];
		                line[j] = 0;
		                i--; // retry this entry
		            } else if (line[i] == line[j]) {
		                if(line[i] != 0xf) {
		                    /* Pretend that 32768 + 32768 = 32768 (representational limit). */
		                    line[i]++;
		                }
		                line[j] = 0;
		            }
		        }

		        row_t result = (line[0] <<  0) |
		                       (line[1] <<  4) |
		                       (line[2] <<  8) |
		                       (line[3] << 12);
		        row_t rev_result = reverse_row(result);
		        unsigned rev_row = reverse_row(row);

		        row_left_table [    row] =                row  ^                result;
		        row_right_table[rev_row] =            rev_row  ^            rev_result;
		        col_up_table   [    row] = unpack_col(    row) ^ unpack_col(    result);
		        col_down_table [rev_row] = unpack_col(rev_row) ^ unpack_col(rev_result);
		    }
	}

	board_t Game2048::execute_move_0(board_t board) {
	    board_t ret = board;
	    board_t t = transpose(board);
	    ret ^= col_up_table[(t >>  0) & ROW_MASK] <<  0;
	    ret ^= col_up_table[(t >> 16) & ROW_MASK] <<  4;
	    ret ^= col_up_table[(t >> 32) & ROW_MASK] <<  8;
	    ret ^= col_up_table[(t >> 48) & ROW_MASK] << 12;
	    return ret;
	}

	board_t Game2048::execute_move_1(board_t board) {
	    board_t ret = board;
	    board_t t = transpose(board);
	    ret ^= col_down_table[(t >>  0) & ROW_MASK] <<  0;
	    ret ^= col_down_table[(t >> 16) & ROW_MASK] <<  4;
	    ret ^= col_down_table[(t >> 32) & ROW_MASK] <<  8;
	    ret ^= col_down_table[(t >> 48) & ROW_MASK] << 12;
	    return ret;
	}

	board_t Game2048::execute_move_2(board_t board) {
	    board_t ret = board;
	    ret ^= board_t(row_left_table[(board >>  0) & ROW_MASK]) <<  0;
	    ret ^= board_t(row_left_table[(board >> 16) & ROW_MASK]) << 16;
	    ret ^= board_t(row_left_table[(board >> 32) & ROW_MASK]) << 32;
	    ret ^= board_t(row_left_table[(board >> 48) & ROW_MASK]) << 48;
	    return ret;
	}

	board_t Game2048::execute_move_3(board_t board) {
	    board_t ret = board;
	    ret ^= board_t(row_right_table[(board >>  0) & ROW_MASK]) <<  0;
	    ret ^= board_t(row_right_table[(board >> 16) & ROW_MASK]) << 16;
	    ret ^= board_t(row_right_table[(board >> 32) & ROW_MASK]) << 32;
	    ret ^= board_t(row_right_table[(board >> 48) & ROW_MASK]) << 48;
	    return ret;
	}

	/* Execute a move. */
	board_t Game2048::execute_move(int move, board_t board) {
	    switch(move) {
	    case 0: // up
	        return execute_move_0(board);
	    case 1: // down
	        return execute_move_1(board);
	    case 2: // left
	        return execute_move_2(board);
	    case 3: // right
	        return execute_move_3(board);
	    default:
	    	throw std::runtime_error("Invalid action, we only support four actions: up(0), down(1), left(2), right(3)!");
	        
	    }
	    return ~0ULL;
	}


	int Game2048::get_max_rank(board_t board) {
	    int maxrank = 0;
	    while (board) {
	        maxrank = std::max(maxrank, int(board & 0xf));
	        board >>= 4;
	    }
	    return maxrank;
	}

	int Game2048::count_distinct_tiles(board_t board) {
	    uint16_t bitset = 0;
	    while (board) {
	        bitset |= 1<<(board & 0xf);
	        board >>= 4;
	    }

	    // Don't count empty tiles.
	    bitset >>= 1;

	    int count = 0;
	    while (bitset) {
	        bitset &= bitset - 1;
	        count++;
	    }
	    return count;
	}



	int Game2048::count_empty(board_t x){
	    x |= (x >> 2) & 0x3333333333333333ULL;
	    x |= (x >> 1);
	    x = ~x & 0x1111111111111111ULL;
	    // At this point each nibble is:
	    //  0 if the original nibble was non-zero
	    //  1 if the original nibble was zero
	    // Next sum them all
	    x += x >> 32;
	    x += x >> 16;
	    x += x >>  8;
	    x += x >>  4; // this can overflow to the next nibble if there were 16 empty positions
	    return x & 0xf;
	}
}
