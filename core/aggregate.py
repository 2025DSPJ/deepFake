import numpy as np

def aggregate_and_decide(raw_conf_for_vote, results, scene_lumas, mode):
    # weights
    weights, scores = [], []
    conf_list = [r['confidence'] for r in results]

    for p, q, l in raw_conf_for_vote:
        w_conf = 0.5 + abs(p - 0.5)
        w_quality = min(1.0, q / 110.0)
        w_light = 0.9 if (l < 40 or l > 210) else 1.0
        w = w_conf * w_quality * w_light
        weights.append(w); scores.append(p)

    S_mean = float(np.dot(weights, scores) / max(np.sum(weights), 1e-6)) if weights else 0.0

    conf_sorted = sorted(conf_list)
    if len(conf_sorted) >= 10:
        k = max(1, int(0.10 * len(conf_sorted)))
        trimmed = conf_sorted[k:len(conf_sorted)-k]
        S_trim = float(np.mean(trimmed)) if trimmed else S_mean
    else:
        S_trim = S_mean

    frac_high = (sum(c >= 0.85 for c in conf_list) / max(1, len(conf_list))) if conf_list else 0.0
    streak3 = any(conf_list[i] >= 0.8 and conf_list[i+1] >= 0.8 and conf_list[i+2] >= 0.8
                  for i in range(0, max(0, len(conf_list)-2)))

    if mode == 'precision':
        tau_low, tau_high, need_frac = 0.50, 0.58, 0.25
    else:
        tau_low, tau_high, need_frac = 0.50, 0.62, 0.35

    S = S_trim
    final_label = 'FAKE' if (S >= tau_high and (frac_high >= need_frac or streak3)) else 'REAL'
    frame_vote_ratio = float(sum(1 for r in results if r['pred'] == 1)) / float(len(results))

    return final_label, S, tau_high, frame_vote_ratio
