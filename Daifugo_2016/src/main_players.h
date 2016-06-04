/*
 * main_players.h
 *
 *  Created on: 2016/05/13
 *      Author: sin
 */

#ifndef MAIN_PLAYERS_H_
#define MAIN_PLAYERS_H_

#include "Player.h"
#include "SamplePlayer.h"

#include "./groups2016/Group1.h"
#include "./groups2016/Group2.h"
#include "./groups2016/Group3.h"
#include "./groups2016/Group4.h"
#include "./groups2016/Group5.h"
#include "./groups2016/Group6.h"
#include "./groups2016/Group7.h"
#include "./extra/ThinkTA1.h"
#include "./groups2013/GroupF.h"
#include "./groups2015/Group7.h"

void registerPlayers(Dealer & d) {
	d.regist(new Group3("Group 3"));
	d.regist(new Group6("Group 6"));
	d.regist(new grp2013::GroupF("Group F/13"));
	d.regist(new grp2015::Group7("Group 7/15"));
	d.regist(new ThinkTA1("ThinkTA1"));
}


#endif /* MAIN_PLAYERS_H_ */
