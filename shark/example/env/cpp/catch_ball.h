#ifndef SHARE_EXAMPLE_ENV_CATCH_BALL_H
#define SHARE_EXAMPLE_ENV_CATCH_BALL_H



#include <utility>
#include <vector>
#include <tuple>

#include "inc/cnarray.hpp"
#include "crandom.h"

struct Color{
    uint8_t r;
    uint8_t g;
    uint8_t b;

    Color(uint8_t tr, uint8_t tg, uint8_t tb)
    :r(tr), g(tg), b(tb){}
};
struct Size {
    int w;
    int h;

    Size(int tw, int th)
    : w(tw), h(th){}
};

class Rect{
public:
    Rect(int l, int t, int w, int h)
    : left_(l), top_(t), width_(w), height_(h)
    {}

    int left() const { return left_; }
    int top() const { return top_; }
    int width() const { return width_; }
    int height() const { return height_; }

    int bottom() const { return top_ - height_; }
    int right() const { return left_ + width_; }

    void move2left(int x){ left_ -= x;  }
    void move2right(int x){ left_ += x; }

    void move2top(int x){ top_ += x; }
    void move2bottom(int x){ top_ -= x; }


    void set_left(int x){ left_ = x; }
    void set_top(int x){ top_ = x; }
protected:
    int left_;
    int top_;
    int width_;
    int height_;
};


typedef cndarray<uint8_t, 3> uint8_3darray;
typedef cndarray<int, 1> int_1darray;


class CatchBallSimulate{

public:
    CatchBallSimulate(const Size & screen, const Size & ball, const Size & ball_speed, const Size & bar, 
             double, int, bool is_continuous);

    void draw(uint8_3darray & screen);

    /// action = 1 move right; action = -1 move left; return reward
    bool step(int action, double & reward);
  
    void reset();
    void reset_ball();
    void seed(int seed_id){rand_.seed(seed_id);}

    std::pair<int, int> action_range()const{
        return action_range_;
    }
    void set_action_range(const std::pair<int, int> & act_range){
        action_range_ = act_range;
    }
    std::pair<int, int> screen_size()const{
        return std::pair<int, int>( screen_size_.h,  screen_size_.w );
    }
    std::pair<int, int> ball_size()const {
        return std::pair<int, int>( ball_size_.h,  ball_size_.w );
    }
    std::pair<int, int> bar_size()const{
        return std::pair<int, int>( bar_size_.h,  bar_size_.w );
    }
    bool is_continuous()const{return is_continuous_;}
protected:
    
    const Size screen_size_;
    const Size ball_size_;
    const Size bar_size_;
    Rect ball_pos_;
    Rect bar_pos_;

    int ball_dir_x_;
    int ball_dir_y_;

    std::pair<int, int> action_range_;

    const int waiting_;
    int cur_waits_;
    const double action_penalty_;
    const bool is_continuous_;

    CRandom rand_;

    void _move_bar(Rect & bar, int action);
    void _move_ball(Rect & ball);
    bool _get_reward(const Rect &bar, double &reward);
    void _draw(const Color & c, const Rect & rect, uint8_3darray & screen);

};


#endif