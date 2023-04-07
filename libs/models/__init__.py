import logging

# from monai.networks.nets import unet,unetr,segresnet,dynunet
# from monai.networks.nets import Unet, UNETR, SegResNet,DynUnet
from monai.networks.nets import Unet, UNETR, SegResNet
# from monai.networks.nets import *

from .mtmaunet import DynUnet



logger = logging.getLogger("weakcls")

key2net = {
    "unet": Unet,
    "unetr":  UNETR,
    "segresnet": SegResNet,
    "mtmaunet": DynUnet,
}


def get_network(cfg):
    if cfg["network"]["type"] is None or  cfg["network"][cfg["network"]["type"]]["name"] is None:
        logger.info("Using Unet as segmentation structure")
        return Unet
    
    else:
        net_name = cfg["network"][cfg["network"]["type"]]["name"]
        if net_name not in key2net:
            raise NotImplementedError("network {} not implemented".format(net_name))

        logger.info("Using {} as network".format(net_name))
        return key2net[net_name]