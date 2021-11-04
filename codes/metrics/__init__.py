from .metric_calculator import MetricCalculator


def create_metric_calculator(opt):
    if 'metric' in opt and opt['metric']:
        return MetricCalculator(opt)
    else:
        return None
