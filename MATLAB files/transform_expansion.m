%transform points based on expanding of the ED to ES. the coefficients come
%from the first set. test on the second set
function [ED_numer, ES_number, x_est,y_est] = transform_expansion (number, area, Patient_ID, Slice_Number , ED_outof20, ES_outof20) %points1 ED (fixed) --> points1 ES (fixed)   points in between?
f1=load('C:\Users\r_beh\OneDrive\Desktop\Freehand_01');
f2=load('C:\Users\r_beh\OneDrive\Desktop\Freehand_02');
freehand_1=f1.freehand_1;
freehand_2=f2.freehand_1;
GT_folder = strcat('C:\Users\r_beh\OneDrive\Desktop\new\new\',Patient_ID,'\contours-manual\IRCCI-expert');
[GTx_ED,GTy_ED] = load_data(strcat(GT_folder,'\IM-0001-',num2str(Slice_Number*20+ED_outof20,'%04.0f') ,'-icontour-manual.txt'));
[GTx_ES,GTy_ES] = load_data(strcat(GT_folder,'\IM-0001-',num2str(Slice_Number*20+ES_outof20,'%04.0f') ,'-icontour-manual.txt'));
ED_area = polyarea(GTx_ED,GTy_ED)
ES_area = polyarea(GTx_ES,GTy_ES)
coeff=(ES_area/ED_area);
%%%%%%%%%%%%area_specific= coeff*area/area(10)
area_specific = (area*coeff)/area(10);
area_specific=area_specific/area_specific(4)
%area_specific= sqrt(area_specific)
%ED and ES slices for the second set 
x_ed = GTx_ED;%(freehand_2(88).Position(:,1));
y_ed = GTy_ED;%(freehand_2(88).Position(:,2));
x_es = GTx_ES;%(freehand_2(100).Position(:,1));
y_es = GTy_ES;%(freehand_2(100).Position(:,2));

[avgx_ed,avgy_ed,A_mat_ed,Y_mat_ed] = angle_generator(x_ed,y_ed);
[avgx_es,avgy_es,A_mat_es,Y_mat_es] = angle_generator(x_es,y_es);
m = (avgy_es-avgy_ed) ./(avgx_es-avgx_ed);
%the moving slice estimation for slice 80+2*number
portion = (area_specific(number)-1)/ (area_specific(floor(ES_outof20/2))-1)*1.25
x_est=avgx_ed + (avgx_es-avgx_ed)*portion; %delta x
%x_est = (area_specific(number)-1)*avgx_ed;

%y_est=avgy_ed+ m .*[(avgx_es-avgx_ed)/(area_specific(10))]*(area_specific(number)); % delta y = m delta x
y_est=avgy_ed+ m .*(avgx_es-avgx_ed)*portion; % delta y = m delta x

for j = 1:150
     if (avgx_ed(j) == avgx_es(j))
    x_est(j) = avgx_ed(j);
    y_est(j) = avgy_ed(j) + (avgy_es(j) -avgy_ed(j)) *portion;
     end
end
%  figure
%  set(gcf, 'Position', get(0, 'Screensize'));
%  plot(x_est,y_est,'r')
%  hold on
% % %hold on
%  plot(x_ed, y_ed)
%  hold on
%  plot(x_es,y_es)
% %hold on
% plot(avgx_ed,avgy_ed)
% hold on
% plot(avgx_es,avgy_es)
 axis equal
 ED_numer = ED_outof20;
 ES_number= ES_outof20;
end

