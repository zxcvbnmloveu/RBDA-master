# RBDA-master
Implementation of reducing bias to source samples for unsupervised domain adaptation (RBDA) by wei

train:

python train_image_MeanTeacher88.4.py --method CDAN+MT+cent+VAT+weightCross+T --s_dset_path ./data/office/webcam_list.txt --t_dset_path ./data/office/amazon_list.txt --save_name DA --gpu_id 0
python train_image_MeanTeacher88.4.py --num_iterations 100000 --test_interval 3000 --method CDAN+MT+cent+VAT+weightCross+T --dset visda --s_dset_path ./data/visda-2017/train_list.txt --t_dset_path ./data/visda-2017/validation_list.txt --save_name visda --gpu_id 1
python train_image_MeanTeacher88.4.py --dset image-clef --method CDAN+MT+cent+VAT+weightCross --s_dset_path ./data/image-clef/list/i_list.txt --t_dset_path ./data/image-clef/list/p_list.txt --save_name _IP --gpu_id 1


result:
![image](https://raw.githubusercontent.com/zxcvbnmloveu/RBDA-master/master/clef.png)
![image](https://raw.githubusercontent.com/zxcvbnmloveu/RBDA-master/master/office31.png)

