__author__ = 'brandonkelly'

from run_multi_component_sampler import main as train
from ppc_multi_component import main as ppc
from predictions_multi_component import main as predict


if __name__ == "__main__":
    train(15, 7500, 2500)
    predict(250)
    ppc(100)
