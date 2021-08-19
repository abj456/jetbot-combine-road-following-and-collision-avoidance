# jetbot combine road following and collision avoidance
If we use the original nvidia model, GPU on jetbot cannot work successfully since it is unable to handle 2 model simultaneously.

By torch2trt library, I transform the original pytorch model to tensorrt model to raise performance, so that we can combine road following, endline parking, and collision avoidance.

However, if the object is in the right hand side, jetbot will crash and keep proceeding. I guess the conditional judgement in last part causes this crash.
