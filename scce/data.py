import os

import fanc
from fanc.compatibility.cooler import CoolerHic
from fanc.compatibility.juicer import JuicerHic


class HiCLoader:
    HiC = ".hic"
    Mcool = ".mcool"
    Cool = ".cool"

    def __init__(self, folder_path, resolution):
        self.folder_path = folder_path
        self.resolution = resolution

    def load_hic(self, file_name):
        _suffix = os.path.splitext(file_name)[-1]

        _file_path = os.path.join(self.folder_path, file_name)
        if _suffix in [HiCLoader.HiC, HiCLoader.Mcool]:
            c = fanc.load("{}@{}".format(_file_path, self.resolution), mode="r")
        elif _suffix == HiCLoader.Cool:
            c = fanc.load(_file_path, mode="r")
        else:
            raise ValueError("File {file_name} is not a valid hic file.")

        return c

    def get_contact(self, c):
        if type(c) is JuicerHic:
            # TODO
            pass
        elif type(c) is CoolerHic:
            contact = c.pixels(join=True)[:]
        return contact

    def catch_matrix(self, c, chrom: str, start: int, end: int, strand: str = None):
        if strand == "+":
            start -= 5000
        elif strand == "-":
            end += 5000

        _loc = "{}:{}-{}".format(chrom, start, end)
        return c.matrix((_loc, _loc)).data
