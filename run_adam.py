from library import TNMLclass
from library import Parameters as Pa

pa = Pa.gtn_net()
A = TNMLclass.GTN_Net(para=pa, device='cuda:0', debug_mode=False)
A.start_learning()