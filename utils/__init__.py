from utils import ramps

def get_current_consistency_weight(epoch, consistency_rampup=200.0):
    """Get the current consistency weight based on the epoch and rampup period."""
    return ramps.sigmoid_rampup(epoch, consistency_rampup)

def patients_to_slices(dataset, patients_num):
    """Convert the number of patients to the corresponding number of slices based on the dataset."""
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "5": 68, "10": 136, "20": 256, "30": 396, "40": 512, "50": 664, "100": 1312}
    elif "Synapse" in dataset:
        ref_dict = {"5": 153, "10": 249, "20": 470, "100": 2211}
    else:
        raise ValueError(patients_num)
    return ref_dict[str(patients_num)]

def update_ema_variables(model, ema_model, alpha, global_step):
    """Update the exponential moving average of the model parameters."""
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
