from promptings.CoT import CoTStrategy
from promptings.Direct import DirectStrategy
from promptings.Analogical import AnalogicalStrategy
from promptings.SelfPlanning import SelfPlanningStrategy

from promptings.MapCoder import MapCoder as MapCoder
from promptings.MapCoder_Ablation import MapCoder_wo_PD, MapCoder_wo_RD, MapCoder_wo_RP, MapCoder_wo_D, MapCoder_wo_P, MapCoder_wo_R


class PromptingFactory:
    @staticmethod
    def get_prompting_class(prompting_name):
        if prompting_name == "CoT":
            return CoTStrategy
        elif prompting_name == "MapCoder":
            return MapCoder
        elif prompting_name == "MapCoder_Ab1":
            return MapCoder_wo_RP
        elif prompting_name == "MapCoder_Ab2":
            return MapCoder_wo_R
        elif prompting_name == "MapCoder_Ab3":
            return MapCoder_wo_RD
        elif prompting_name == "MapCoder_Ab4":
            return MapCoder_wo_P
        elif prompting_name == "MapCoder_Ab5":
            return MapCoder_wo_D
        elif prompting_name == "MapCoder_Ab6":
            return MapCoder_wo_PD
        elif prompting_name == "Direct":
            return DirectStrategy
        elif prompting_name == "Analogical":
            return AnalogicalStrategy
        elif prompting_name == "SelfPlanning":
            return SelfPlanningStrategy
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
