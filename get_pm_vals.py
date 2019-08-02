from cmsaa import pm_summary,pm_summary_gmonly, pm_summary_constantsm, pm_summary_inhibitedgm
from cmsaa import GoalMap, SaliencyMap, PriorityMap
import numpy as np

x = np.arange(-90,90)
#GLFree: -90
#vals = pm_summary(-90,x,0.7645,50,0.7586,51.0609)
#GLFree: 0
#vals = pm_summary(0,x,0.7708,50,0.8000,52.9654)
#GLFree: 90
#vals = pm_summary(0,x,0.7696,50,0.7607,50.5184)

#GMOnly: -90
#vals = pm_summary_gmonly(-90,x,0.8000,200.0000)
#GMOnly: 0
#vals = pm_summary_gmonly(0,x,0.7869,200.0000)
#GMOnly: 90
#vals = pm_summary_gmonly(90,x,0.8000,200.0000)

#ConstantSM: -90
#vals = pm_summary_constantsm(-90,x,0.3000,200,0.4808)
#ConstantSM: 0
#vals = pm_summary_constantsm(0,x,0.3000,200,0.4735)
#ConstantSM: 90
#vals = pm_summary_constantsm(90,x,0.3000,200,0.5)

#InhibitedGM: -90
#vals = pm_summary_inhibitedgm(-90,x,0.3000,118.8839,0.5000,172.7045,0.4662)
#InhibitedGM: 0
#vals = pm_summary_inhibitedgm(0,x,0.3791,50.0021,0.4377,59.6354,0.3921)
#InhibitedGM: 90
#vals = pm_summary_inhibitedgm(90,x,0.4061,50,0.3973,50.8309,0.3595)


print(vals)
