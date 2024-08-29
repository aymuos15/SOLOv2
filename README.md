FOR Evaluation
1. Comment out line 9 and change the loss function from the minimal implementation in solov2_head.py
2. change the loss in line 310 after commenting out the init loss in solov2_head.py
2. Change np.bool to just bool in eval.py in line 188
3. Change the save path in eval.py and the save=True

FOR Training
1. in mask_feat_head.py, Make the following change:
            # feature_add_all_level += self.convs_all_levels[i](input_p)
            feature_add_all_level = self.convs_all_levels[i](input_p) + feature_add_all_level


Original REPO: https://github.com/OpenFirework/pytorch_solov2
