
% example of how to call gbvs with default params
file_names = dir("*.png");
%% 
num_imgs = length(file_names);
gbvs_res = zeros(num_imgs, 32, 32);
for i = 1:num_imgs
    f = file_names(i);
    img = imread(strcat(f.folder, "\", f.name));
    disp(size(img));
    out_gbvs = gbvs(img);
    gbvs_res(i, :, :) = out_gbvs.master_map;
end
save('gbvs_map_popout.mat', 'gbvs_res');