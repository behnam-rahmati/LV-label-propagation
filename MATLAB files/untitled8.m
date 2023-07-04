close all
max_its = 50;
lengthEweight = 10;
shapeEweight = 0.2;
Patient_ID = 'SC-HF-I-01';
slice_number=4;
for i=[2,4,6,8,10,12,14,16,18,20]
    figure
    %number=120+2*i
number2= slice_number*20 +i    
[ED_number,ES_number, expansion_estimation_x(number2,:),expansion_estimation_y(number2,:)] = transform_expansion(i/2,area, Patient_ID,slice_number,8,19);
name = strcat('C:\Users\r_beh\OneDrive\Desktop\Sunnybrook Cardiac MR Database DICOMPart3\Sunnybrook Cardiac MR Database DICOMPart3\TrainingDataDICOM\',Patient_ID,'\DICOM\IM-0001-',num2str(number2,'%04.0f'),'.dcm');
a=dicomread(name);
%figure
init=(poly2mask(expansion_estimation_x(number2,:),expansion_estimation_y(number2,:),256,256));
[seg,phi] = chan_vese(a,init,max_its,lengthEweight,shapeEweight,1);
M = contour(phi, [0 0], 'g','LineWidth',1);
%   imshow(a,'initialmagnification',200,'displayrange',[0 255]); hold on;

  %contour(phi, [0 0], 'k','LineWidth',2);
  %hold off; title([num2str(i) ' Iterations']); 
 
  imshow(a,'initialmagnification',200,'displayrange',[0 255]); 
 hold on
    plot(M(1,2:end), M(2,2:end),'g')
hold on
plot(expansion_estimation_x(number2,:),expansion_estimation_y(number2,:),'r')
hold on
if (i ==ED_number || i==ES_number)
path =strcat("C:\Users\r_beh\data\backup\", Patient_ID,"\contours-manual\IRCCI-expert\",'IM-0001-',num2str(slice_number*20 +i,'%04.0f') ,'-icontour-manual.txt')
[GTx,GTy] = load_data(path);    hold on
    plot(GTx,GTy,'w')





end
end

  
