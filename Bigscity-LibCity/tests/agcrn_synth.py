import torch, sys, traceback
sys.path.insert(0, '.')
try:
    from libcity.model.traffic_speed_prediction.AGCRN import AGCRN
    print("IMPORT OK", AGCRN)
except Exception:
    traceback.print_exc(); sys.exit(1)

class Cfg(dict):
    def __getitem__(self, k):
        return dict.__getitem__(self, k)
    def get(self, k, d=None):
        return dict.get(self, k, d)

cfg = Cfg({'input_window':12,'output_window':12,'rnn_units':64,'embed_dim':10,
           'num_layers':2,'cheb_order':2,'input_dim':1,'output_dim':1,
           'device':torch.device('cpu'),'batch_size':2,'seed':0})
data_feature = {'num_nodes':20,'feature_dim':1,'output_dim':1,'scaler':None}
m = AGCRN(cfg, data_feature)
print("PARAMS", sum(p.numel() for p in m.parameters()))
x = torch.randn(2,12,20,1)
out = m({'X': x})
print("X", tuple(x.shape), "OUT", tuple(out.shape))
assert tuple(out.shape) == (2,12,20,1), out.shape
out.sum().backward()
print("BACKWARD OK")
print("SMOKE TEST PASS")
