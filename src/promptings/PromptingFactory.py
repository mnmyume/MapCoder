from promptings.CoT import CoTStrategy
from promptings.Direct import DirectStrategy
from promptings.Analogical import AnalogicalStrategy
from promptings.SelfPlanning import SelfPlanningStrategy

from promptings.MapCoder import MapCoder as MapCoder
from promptings.MapCoderMAS import MapCoderMAS as MapCoderMAS


class PromptingFactory:
    @staticmethod
    def get_prompting_class(prompting_name):
        if prompting_name == "CoT":
            return CoTStrategy
        elif prompting_name == "MapCoder":
            return MapCoder
        elif prompting_name == "MapCoderMAS":
            return MapCoderMAS
        elif prompting_name == "Direct":
            return DirectStrategy
        elif prompting_name == "Analogical":
            return AnalogicalStrategy
        elif prompting_name == "SelfPlanning":
            return SelfPlanningStrategy
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")

