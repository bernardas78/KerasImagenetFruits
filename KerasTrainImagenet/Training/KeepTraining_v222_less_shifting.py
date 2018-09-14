# Keep training model v221, but data shifts 5% (instead of former 10%). 
#   Expecting to increase train accuracy

from Evaluation import Eval_v1_simple as e_v1
#from Evaluation import Eval_v2_top5accuracy as e_v2

dataGen = dg_v2.prepDataGen(shift_range=0.05)

epochs = 50

eval_every_epochs = 10

for eval_period in range(int(epochs/eval_every_epochs)):
    print ("eval_period:", str(eval_period) )
    model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=eval_every_epochs, verbose=1 )
    e_v1.eval (model)
