#### Based on data2.0.2
#### add code ' sub_obj_overlap_all_train, sub_obj_overlap_all_dev = self._valid_sub_obj_intersection()
        for item in sub_obj_overlap_all_dev:
            if item not in sub_obj_overlap_all_train:
                sub_obj_overlap_all_train.add(item)' 
#### add code 'if s in sub_obj_overlap_all_global or o in sub_obj_overlap_all_global:' in the 3 strategies.
#### The goal is to reduce the amount of 'other'.
#### used in '/home/zutnlp/zhangtao/lic2019/Information_Extraction/information-extraction-baseline/relationship_extraction/entity-aware-relation-classification_50_relationship/data/update_20190423'