import torch
import torch.nn as nn
from NGCF.NGCF import NGCF


class Trainer(object):
    def __init__(self, model, args):
        self.model = model

        """
        print(self.model.L.type())
        print(self.model.L.get_device())

        print(self.model.L_I.type())
        print(self.model.L_I.get_device())
        
        for m in self.model.children():
            print(m)
        """

        print("model2half")

        self.model.L = self.model.L.half()
        self.model.L_I = self.model.L_I.half()
        self.model = self.network_to_half(self.model)


        """
        for module in self.model.modules():
            if isinstance(module, NGCF):
                print(module.L.type())
                print(module.L.get_device())
                print(module.L_I.type())
                print(module.L_I.get_device())
        
        for m in self.model.children():
            print(m)
        """

    def network_to_half(self, model):
        """
        Convert model to half precision
        Access list :
                        1) User, item embeddings
                        2) Adj matrix
                        3) Networks weights
        """

        return nn.Sequential(model.half())

    def convert_float(self, module):
        print(module.modules())
    # User, item ID 값들이 input으로 들어가기 때문에 half로 바꿀 필요가 없음
    # class tofp16(nn.Module):
    #     def __init__(self):
    #         super(Trainer.tofp16, self).__init__()
    #
    #     def forward(self, input):
    #         return input.half()
