from scipy.optimize import linear_sum_assignment
import numpy as np

   
class doa_sde_metric:
    def __init__(self):
        self._eps = 1e-7

        self._localization_error = 0
        self._distance_error = 0
        self._tp_doa = 0
        self._tp_sde = 0
        self._accuracy = 0
        return

    def partial_compute_metric(self, dist_mat, dist_diff, is_greater, gt_activity, pred_activity=None, max_doas=1):
        self._distance_error += np.sum(dist_diff)
        self._accuracy += np.sum(is_greater)
        self._tp_sde += len(dist_diff)

        
        for frame_cnt, loc_dist in enumerate(dist_mat):
            nb_active_gt = int(gt_activity[frame_cnt].sum())
            nb_active_pred = max_doas if pred_activity is None else int(pred_activity[frame_cnt].sum()) #TODO remove hard coded max value of 2 DoAs 
            self._tp_doa += np.min((nb_active_gt, nb_active_pred))

            if nb_active_gt and nb_active_pred:
                if pred_activity is None:
                    if nb_active_gt==1:
                        loc_dist = loc_dist[:, 0][None]
                else:
                    loc_dist = loc_dist[pred_activity[frame_cnt]==1, :][:, gt_activity[frame_cnt]==1]
                row_ind, col_ind = linear_sum_assignment(loc_dist)
                loc_err = loc_dist[row_ind, col_ind].sum()

                self._localization_error += loc_err
        return

    def get_results(self):
        LE = self._localization_error/self._tp_doa
        DE = self._distance_error/self._tp_sde
        ACC = 100.*self._accuracy/self._tp_sde
        return 180.*LE/np.pi, DE, ACC

class doa_sde_metric_bin:
    def __init__(self):
        self._eps = 1e-7

        self._localization_error = 0
        self._tp_doa = 0
        self._tp_sde = 0
        self._accuracy = 0
        return

    def partial_compute_metric(self, dist_mat, is_same, gt_activity, pred_activity=None, max_doas=1):
        if len(np.shape(is_same)) > 1:
            is_same = is_same.flatten()
        self._accuracy += np.sum(is_same)
        self._tp_sde += len(is_same)

        
        for frame_cnt, loc_dist in enumerate(dist_mat):
            nb_active_gt = int(gt_activity[frame_cnt].sum())
            nb_active_pred = max_doas if pred_activity is None else int(pred_activity[frame_cnt].sum()) #TODO remove hard coded max value of 2 DoAs 
            self._tp_doa += np.min((nb_active_gt, nb_active_pred))

            if nb_active_gt and nb_active_pred:
                if pred_activity is None:
                    if nb_active_gt==1:
                        loc_dist = loc_dist[:, 0][None]
                else:
                    loc_dist = loc_dist[pred_activity[frame_cnt]==1, :][:, gt_activity[frame_cnt]==1]
                row_ind, col_ind = linear_sum_assignment(loc_dist)
                loc_err = loc_dist[row_ind, col_ind].sum()

                self._localization_error += loc_err
        return

    def get_results(self):
        LE = self._localization_error/self._tp_doa
        ACC = 100.*self._accuracy/self._tp_sde
        return 180.*LE/np.pi, 0., ACC