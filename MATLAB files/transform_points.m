function [x2es_est,y2es_est] = transform_points (x1ed,y1ed,x1es,y1es,x2ed,y2ed,x2es,y2es, number) %points1 ED --> points1 ES       points2 ED -->? points2 ES
[avgx1_ed,avgy1_ed,A1_mat_ed,Y1_mat_ed] = angle_generator(x1ed,y1ed);
[avgx1_es,avgy1_es,A1_mat_es,Y1_mat_es] = angle_generator(x1es,y1es);
[avgx2_ed,avgy2_ed,A2_mat_ed,Y2_mat_ed] = angle_generator(x2ed,y2ed);
[avgx2_es,avgy2_es,A2_mat_es,Y2_mat_es] = angle_generator(x2es,y2es);
P = A1_mat_ed \Y1_mat_es;
Y2_es_estimated = A2_mat_ed * P;
x2es_est = Y2_es_estimated(:,1);
y2es_est = Y2_es_estimated(:,2);
fh1 = figure();
fh1.WindowState = 'maximized';
name = strcat('C:\Users\r_beh\cardiac-segmentation-master\cardiac-segmentation-master\cardiac-segmentation-master\Sunnybrook_data\challenge_training\SC-HF-I-01\DICOM\IM-0001-',num2str(number+88,'%04.0f'),'.dcm');
a=dicomread(name);
imshow(imadjust(a));
hold on
plot(x2es_est, y2es_est,'r')
hold on 
plot(x2es,y2es,'g')
%hold on
%plot(x2ed, y2ed,'b')
axis equal
title('second set result')
fh2 = figure();
fh2.WindowState = 'maximized';
name = strcat('C:\Users\r_beh\cardiac-segmentation-master\cardiac-segmentation-master\cardiac-segmentation-master\Sunnybrook_data\challenge_training\SC-HF-I-01\DICOM\IM-0001-',num2str(number+68,'%04.0f'),'.dcm');
a=dicomread(name);
imshow(imadjust(a));
hold on
plot(x1ed, y1ed,'b')
hold on
plot(x1es, y1es,'r')
title('first set ED to the shifting slice')
axis equal
