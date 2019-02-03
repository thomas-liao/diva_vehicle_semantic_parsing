"""Thomas Liao - Sep 2018"""

import tensorflow as tf

import hg_utils as ut

class hier_loss:
    def __init__(self):
        self.hierarchical_structure = {}
        # hierarchical level-1: whole car
        self.hierarchical_structure['lvl_1'] = {'whole': [i for i in range(36)]}
        
        # hierarchical level -2: left, right, front, end, top, bottom - 6 parts
        front_indices = [0, 1, 2, 3, 12, 13, 14, 15, 16]
        front_indices = front_indices + [i + 18 for i in front_indices]
        rear_indices = [i for i in range(4, 12)]
        rear_indices.append(17)
        rear_indices = rear_indices + [i + 18 for i in rear_indices]
        right_indices = [i for i in range(18)]
        left_indices = [i + 18 for i in right_indices]
        
        top_indices = [i for i in range(1, 7)]
        top_indices.extend([i + 18 for i in top_indices])
        bottom_indices = [0] + [i for i in range(7, 18)] + [18] + [i + 18 for i in range(7, 18)]
        
        self.hierarchical_structure['lvl_2'] = {'front': front_indices, 'rear': rear_indices, 'left': left_indices,
                                        'right': right_indices, 'top': top_indices, 'bottom': bottom_indices}
        
        # hierarchical level - 3: small parts: front/rear - left/right wheels (4 parts), left/right front-rear panels (4 parts),
        # left/right - front-rear roofs ( 4 parts), total 12 parts
        
        # wheels
        front_right_wheel_indices = [i for i in range(12, 17)]
        front_left_wheel_indices = [i+18 for i in front_right_wheel_indices]
        rear_right_wheel_indices = [8, 9, 10, 11, 17]
        rear_left_wheel_indices = [i+18 for i in rear_right_wheel_indices]
        
        # panels
        front_right_panel_indices = [0, 1, 2]
        front_left_panel_indices = [i+18 for i in front_right_panel_indices]
        rear_right_panel_indices = [5, 6, 7]
        rear_left_panel_indices = [i+18 for i in rear_right_panel_indices]
        
        # roofs
        front_right_roof_indices = [2, 3]
        front_left_roof_indices = [20, 21]
        rear_right_roof_indices = [4, 5]
        rear_left_roof_indices = [22, 23]
        self.hierarchical_structure['lvl_3'] = {'frw':front_right_wheel_indices, 'flw': front_left_wheel_indices, 'rrw': rear_right_wheel_indices, 'rlw': rear_left_wheel_indices,
                                        'frp': front_right_panel_indices, 'flp': front_left_panel_indices, 'rrp': rear_right_panel_indices, 'rlp': rear_left_panel_indices,
                                        'frr': front_right_roof_indices, 'flr': front_left_roof_indices, 'rrr': rear_right_roof_indices, 'rlr': rear_left_roof_indices}
    
    def hier_structure_loss(self, p_hm, gt_hm, weights):
        """Hierarchical loss - tensorflow implementation
        Args:
            p_hm, gt_hm: predict, groundtruthheat map, each with shape of [batch_size, num_stacks(hg), heat_map_height, heat_map_width, num_key_points]
            weights: shape (4,), weights for level-1 (whole_car), level-2(6 parts), level-3 (12 parts), level-4 (36 individual key points)
        Returns:
            (tf.Tensor) hierarchical loss w.r.t. 4 structure level weighted by given weights
        """
        # level-1 loss:
        loss_lvl_1 = weights[0] * self.group_loss_helper(p_hm, gt_hm, self.hierarchical_structure['lvl_1']['whole'])

        # level-2 loss:
        loss_lvl_2 = 0.0
        keys_lvl_2 = self.hierarchical_structure['lvl_2'].keys()
        for k in keys_lvl_2:
            loss_lvl_2 += self.group_loss_helper(p_hm, gt_hm, self.hierarchical_structure['lvl_2'][k])
        loss_lvl_2 = weights[1] * (1.0 / 6) * loss_lvl_2

        # level-3 loss:
        loss_lvl_3 = 0.0
        keys_lvl_3 = self.hierarchical_structure['lvl_3'].keys()
        for k in keys_lvl_3:
            loss_lvl_3 += self.group_loss_helper(p_hm, gt_hm, self.hierarchical_structure['lvl_3'][k])
        loss_lvl_3 = weights[2] * (1.0 / 12) * loss_lvl_3

        # level-4 loss:
        loss_lvl_4 = weights[3] *  ut._bce_loss(logits=p_hm, gtMaps=gt_hm, name='lvl4_ce_loss', weighted=False)
        # loss_lvl_4 = weights[3] * tf.losses.mean_squared_error(predictions=p_hm, labels=gt_hm)

        total_loss = loss_lvl_1 + loss_lvl_2 + loss_lvl_3 + loss_lvl_4
        return total_loss

    def group_loss_helper(self, p_hm, gt_hm, group_idx):  ## tensorflow sucks.......
        p_hm_group = tf.gather(p_hm, tf.convert_to_tensor(group_idx), axis=tf.constant(-1, dtype=tf.int32))
        p_hm_group = tf.reduce_sum(p_hm_group, axis=-1)
        gt_hm_group = tf.gather(gt_hm, tf.convert_to_tensor(group_idx), axis=tf.constant(-1, dtype=tf.int32))
        gt_hm_group = tf.reduce_sum(gt_hm_group, axis=-1)
        # return ut._bce_loss(logits=p_hm_group, gtMaps=gt_hm_group, weighted=False) # give negative number right now... TODO: debug
        return tf.losses.mean_squared_error(predictions=p_hm_group, labels=gt_hm_group) # MSE



#     keys = self.hierarchical_structure.keys()
# for i in range(len(keys)):
#     print(keys[i], self.hierarchical_structure[keys[i]].keys())
