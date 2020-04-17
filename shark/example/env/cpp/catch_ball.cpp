#include "catch_ball.h"

#include <iostream>
#include <ctime>


CatchBallSimulate::CatchBallSimulate(const Size & screen, const Size & ball, const Size & ball_speed,  const Size & bar, 
			 double action_penalty, int waiting, bool is_continuous)
    : screen_size_(screen), ball_size_(ball), bar_size_(bar)
    , ball_pos_ (screen.w / 2 - ball.w / 2,  screen.h/2 - ball.h/2,  ball.w, ball.h)
    , bar_pos_( screen_size_.w / 2 - bar_size_.w / 2, screen_size_.h, bar_size_.w, bar_size_.h )

    , ball_dir_x_( ball_speed.w ), ball_dir_y_( ball_speed.h )
    , action_range_(-20, 20)
    , waiting_(waiting), cur_waits_(0)
    , action_penalty_(action_penalty)
    , is_continuous_(is_continuous)

    , rand_(0, screen_size_.w - ball_size_.w, time(NULL))
    {    }


void CatchBallSimulate::draw(uint8_3darray & screen){
	const Color white(255, 255, 255);

    _draw(white, bar_pos_, screen);
    if(cur_waits_ >= waiting_){
		_draw(white, ball_pos_, screen);
	}
}
	
void CatchBallSimulate::_draw(const Color & c, const Rect & rect,  uint8_3darray & screen){
    int l = rect.left(), r = rect.right(),
        b = rect.bottom(), t = rect.top();

    if(l <= 0)    l = 1;
    if(r >= screen_size_.w) r = screen_size_.w;

    if(b <= 0)    b = 1;
    if(t >= screen_size_.h) t = screen_size_.h;
	for(int j = b-1; j < t; ++j){
		for(int i = l-1; i < r; ++i){
            screen.ix(j, i, 0) = c.r;
            screen.ix(j, i, 1) = c.g;
            screen.ix(j, i, 2) = c.b;
        }
    }
}



bool CatchBallSimulate::_get_reward(const Rect &bar, double & reward){

    if( bar.bottom() <= ball_pos_.top() ){
        if( bar.left() <= ball_pos_.right() && bar.right() >= ball_pos_.left() ){
            if(bar.right() < ball_pos_.right()){
                reward = bar.right() - ball_pos_.left();
            }else if(bar.left() > ball_pos_.left()){
                reward = ball_pos_.right() - bar.left();
            }else{
                reward = ball_pos_.right() - ball_pos_.left();
            }
            reward /= ball_pos_.width();
            reward = reward >= .5 ? 1.0 : 0;
        }
        
        return true;
    }
    return false;
}

void CatchBallSimulate::reset_ball(){
	ball_pos_.set_top(ball_size_.h/2);
	ball_pos_.set_left( rand_.sample()  );
}



void CatchBallSimulate::reset(){
///	rng_.seed( time(NULL) + ((++ seed_) % 100000000) );
///	gen_.engine().seed( time(NULL)  + ((++ seed_) % 100000000) );
///	gen_.distribution().reset();
	
	reset_ball();
    const double ratio =  (double(screen_size_.w - bar_size_.w)) / (screen_size_.w - ball_size_.w) ;

    int l = int( rand_.sample() * ratio);
    bar_pos_.set_left( l );

}



void CatchBallSimulate::_move_bar(Rect & bar, int action){
    if(is_continuous_){
        if(action > action_range_.second) action = action_range_.second;
        else if(action < action_range_.first) action = action_range_.first;
    }else{
        /// in discrete setting, receive action value 0, 1, 2
        action -= 1;
        if(action > 0) action = 5;
        else if(action < 0) action = -5;
    }

    int bar_left = bar.left();
    bar_left += action;
    if(bar_left < 0){
        bar_left = 0;
    }else if(bar_left > screen_size_.w - bar_size_.w){
        bar_left = screen_size_.w - bar_size_.w;
    }
    bar.set_left(bar_left);
}


void CatchBallSimulate::_move_ball(Rect & ball){
    if(cur_waits_ > waiting_){
    //  ball_pos_.move2right( ball_dir_x_ );
        ball_pos_.move2top( ball_dir_y_ );
    }else{
        ++cur_waits_;
    }
}



bool CatchBallSimulate::step(int action, double & reward){
    _move_ball(ball_pos_);

    //// move bar
    _move_bar(bar_pos_, action);
    
    bool is_game_over = _get_reward(bar_pos_, reward);
    if(0 != action) reward -= action_penalty_;
    
    if(is_game_over) cur_waits_ = 0;
    
	return is_game_over;
}



