import numpy as np

def calc_apd_s(times_s, vms, repol_thresh=0.9, active_thresh=-10.0, activation_time=None, min_apd_s=0.100,
               ap_amp_resolve_thresh=135):

    if len(times_s) != len(vms):
        raise ValueError("Times array and Vms array are not of equal lengths")
    elif len(times_s) == 0 or len(vms) == 0:
        raise ValueError("Times array or Vms array of length zero")

    if activation_time is not None:
        if not isinstance(activation_time, float):
            raise ValueError(f"{activation_time=} in calc_apd_s is not of type float")

    min_vm, max_vm = np.min(vms), np.max(vms)
    ap_amp = max_vm - min_vm

    if ap_amp >= ap_amp_resolve_thresh:  # Fixed repolarisation threshold for erroneously large AP amps
        vm_repolarised = -80
    else:
        vm_repolarised = min_vm + ap_amp * (1 - repol_thresh)

    # If no activation time is given, we want to compute the activation time
    if activation_time is None:
        # Find the first index where vm >= active_thresh
        #i_activated = np.argmax(vms >= active_thresh)
        #activation_time = times_s[i_activated]

        # Compute when cell was activated based on active_thresh
        for i, vm in enumerate(vms):
            if vm >= active_thresh:
                i_activated = i
                activation_time = times_s[i_activated]
                break

    if activation_time is None:
        raise Exception("Failed to find activation in this cell")

    apd, repolarisation_time = None, None

    # Analyse Vms for repolarisation, after activation only and some min APD buffer
    idxs_after_activation = np.where(times_s >= activation_time + min_apd_s)[0]

    times_after_activation, vms_after_activation = times_s[idxs_after_activation], vms[idxs_after_activation]

    # Now find where the cell repolarised
    for i, vm in enumerate(vms_after_activation):
        if vm <= vm_repolarised:
            # Considers AP to be the time diff between max Vm and Vm 90% repolarised
            repolarisation_time = times_after_activation[i]
            apd = repolarisation_time - activation_time
            break

    return apd, activation_time, repolarisation_time, ap_amp