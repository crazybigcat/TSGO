from library import TNMLclass
from library import Parameters as Pa

pa = Pa.gtn()
A = TNMLclass.GTN(para=pa, device='cuda:0', debug_mode=False)
A.start_learning()