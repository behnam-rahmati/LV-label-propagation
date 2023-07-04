%plot curve evolution
close all
f1=load('C:\Users\r_beh\OneDrive\Desktop\Freehand_01');
f2=load('C:\Users\r_beh\OneDrive\Desktop\Freehand_02');
freehand_1=f1.freehand_1;
freehand_2=f2.freehand_1;
name = strcat('C:\Users\r_beh\cardiac-segmentation-master\cardiac-segmentation-master\cardiac-segmentation-master\Sunnybrook_data\challenge_training\SC-HF-I-01\DICOM\IM-0001-',num2str(68,'%04.0f'),'.dcm');
a=dicomread(name);
figure
set(gcf, 'Position', get(0, 'Screensize'));

%imshow(imadjust(a));
for i= 62:2:80;
    i
    n=(i-62)/2+1;


hold on
    w = waitforbuttonpress;
    if(w)
       if (rem(i,4)==0)
            color = 'r'
        else
            color = 'b'
        end
        plot(freehand_1(i).Position(:,1),freehand_1(i).Position(:,2),'r'); 
        area(n)= polyarea(freehand_1(i).Position(:,1),freehand_1(i).Position(:,2));
        if (i>62)
            hold on
            plot(freehand_1(i-2).Position(:,1),freehand_1(i-2).Position(:,2),'g'); 
        end
        axis equal
        hold on
    end
end
area=area./area(4)*1.01
figure
plot(area)
