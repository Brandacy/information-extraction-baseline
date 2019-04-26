#### based on sub_obj_bio_7labels_2.0, 
#### modify:
#### Add new code 'intersection_sub_obj_train_set, intersection_sub_obj_dev_set = self._valid_sub_obj_intersection()' in get_sub_obj_bio(self, output_path) method. 
#### Add new code         
#### for item in intersection_sub_obj_dev_set:
####     if item not in intersection_sub_obj_train_set:
####             intersection_sub_obj_train_set.add(item)
#### The purpose is to eliminate the ambiguity between subject and object in the NER TASK.