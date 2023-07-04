function [avgx,avgy,A_mat,Y_mat] = angle_generator(x,y)
xc= mean(x);
yc= mean(y);
m= (y-yc)./(x-xc);
angle_tmp = rad2deg(atan(m));
angle = angle_tmp;
angle(x<=xc)=angle_tmp(x<=xc)+180;
angle (x>xc & angle_tmp<0) = angle_tmp ( x>xc & angle_tmp<0)+360;

for ia = 1:150
tmp = find (2.4*(ia-1)<angle & angle <2.4*ia);
x_angles(ia,1:length(tmp))= tmp;
end

for i3=1:150
temp2 = x_angles(i3,:);
tmpp2 = x(temp2(temp2>0));
tmpp2y = y(temp2(temp2>0));
avgx(i3) = mean(tmpp2);
avgy(i3) = mean(tmpp2y);
end
while (sum(isnan(avgx))> 0)
if(isnan(avgx(1)));
    avgx(1)=avgx(2); 
    avgy(1)=avgy(2); 

end
index = find(isnan(avgx));
if(index>1)
avgx(index) = avgx(index-1);
avgy(index) = avgy(index-1);
else
    avgx(index) = avgx(index+1);
    avgy(index)=avgy(index+1);
end
end
%avgx=avgx-xc;
%avgy=avgy-yc;
%generate_A_mat
avgx=avgx';
avgy=avgy';
X12=avgx.^2;
Y12=avgy.^2;
X13=avgx.^3;
Y13=avgy.^3;
X1Y1=avgx.*avgy;
X1Y12=avgx.*avgy.^2;
X12Y1= avgx.^2.*avgy;
X12Y12=avgx.^2.*avgy.^2;
A_mat = [ones(length(avgx),1),avgx,X12,avgy,Y12,X1Y1,X1Y12,X12Y1,X12Y12];
%A_mat = [ones(length(avgx),1),avgx,avgy,X1Y1];


Y_mat = [avgx,avgy];

end
