from src.rag_moe.experts.calibration import CalibrationResidualExpert
from src.rag_moe.experts.itsc import ITSCExpert
from src.rag_moe.experts.itsc_segment_gate import ITSCSegmentGateExpert
from src.rag_moe.experts.raft import RAFTExpert
from src.rag_moe.experts.source_window import SourceWindowExpert
from src.rag_moe.experts.volatility_peak import VolatilityPeakExpert


EXPERT_REGISTRY = {
    "calibration": CalibrationResidualExpert,
    "itsc": ITSCExpert,
    "itsc_segment_gate": ITSCSegmentGateExpert,
    "raft": RAFTExpert,
    "source_window": SourceWindowExpert,
    "volatility_peak": VolatilityPeakExpert,
}


def get_expert_class(name):
    try:
        return EXPERT_REGISTRY[name]
    except KeyError:
        raise KeyError("unknown expert %r" % (name,))


def build_experts(names, configs=None, data_context=None):
    configs = configs or {}
    experts = []
    for name in names:
        expert_class = get_expert_class(name)
        expert = expert_class()
        expert.prepare(configs.get(name, {}), data_context or {})
        expert.freeze()
        experts.append(expert)
    return experts
