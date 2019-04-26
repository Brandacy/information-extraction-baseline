#### based on sub_obj_bio_7labels_1.0, 
#### modify:
#### spo['subject'] =====> str(spo['subject']).replace(" ", '').replace("《", '').replace("》", '')
#### spo['object'] =====> str(spo['object']).replace(" ", '').replace("《", '').replace("》", '')