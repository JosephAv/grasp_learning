close all
clear all
clc
% 
% v1 = [0.309017,0.951057,-5.82354e-17,1.89218e-17]
% v2 = [0.727149,0.686481,-4.20348e-17,4.4525e-17]
% v3 = [0.970533,-0.240968,1.4755e-17,5.9428e-17]
% 
% 
% v11 = circshift(v1,-1)
% 
% m1 = quat2rotm(v11)
% 
% rotz(90)

%pre = [0.5,0.05,0.05,-0.0,0.02,0.055];
pre = [0.05,0.05,0.03,-0.0,0.02,0.055];

tmp = [0.01 : 0.02 : 0.1];
V11 = [];
V1 = [];
for i = 0 : 10 : 359
    for ii = 0 : 10 : 359
        for iii = 0 : 3 : 359
            for j = 1 %: size(tmp,2)
                m1 = rotx(179)*rotz(deg2rad(i))*roty(deg2rad(ii))*rotz(deg2rad(iii));
                v11 = rotm2quat(m1); v11 = v11/norm(v11);
                %pre(1) = tmp(j);
                pre(1) = 0.05;
                V11 = [V11; [pre v11]];
                v1 = [v11(2:4) v11(1)];
                %v1 = circshift(v11,1)
                V1 = [V1; [pre v1]];
                disp([num2str([i ii iii])])
            end
        end
    end
end


csvwrite('csvlist.csv',V1)