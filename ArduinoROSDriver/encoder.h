#ifndef _ENCODER_H_
#define _ENCODER_H_
#include"config.h"
#include <Encoder.h>

 float getWheelRotatialSpeed(Encoder *enc,long &lastEnc,unsigned long last_time,bool reverse){
    unsigned long curt=millis();
    long curenc=enc->read();
    curenc=reverse?-curenc:curenc;
    float w=(curenc-lastEnc)*resolution/((curt-last_time)/1000.0)/r;
    lastEnc=curenc;
    return w;
    
  }

#endif
