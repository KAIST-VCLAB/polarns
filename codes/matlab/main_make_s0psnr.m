path = 'path for raw images';
path_output = '..\..\PolarNS\histogram\object_s0psnr.mat';

dir_imgs = dir(path);
angles = -1:0.01:1;
angles = angles.*pi;

mean_mse = [];
mean_mse_est = [];

for dir_iter = 3:numel(dir_imgs)
    if ~dir_imgs(dir_iter).isdir
        continue
    end

    path_now = fullfile(path,dir_imgs(dir_iter).name);
    if ~isfile(fullfile(path_now,'img_stat.mat'))
        continue
    end
    
%     img_entry = dir(path_now);
%     img_entry = struct2cell(img_entry);
%     img_entry = img_entry(1,:)';
%     img_entry = convertCharsToStrings(img_entry);
%     sum(startsWith(img_entry,'img_gt_')&endsWith(img_entry,'.png'));
    
    loaded = load(fullfile(path_now,'img_stat.mat'));
    img_count = loaded.image_count;
    img_mean = loaded.img_mean ./ (2.^12);
    img_variance = loaded.img_variance ./ (2.^24);
    [img_mean0,img_mean45,img_mean90,img_mean135] = raw2polar(img_mean);
    img_mean_s0 = img_mean0+img_mean45+img_mean90+img_mean135;
    img_mean_s0 = img_mean_s0/2;
    img_mean_s1 = img_mean0-img_mean90;
    img_mean_s2 = img_mean45-img_mean135;
%     figure;imshow_from_raw(img_mean_s0/2);
    [img_variance0,img_variance45,img_variance90,img_variance135] = raw2polar(img_variance);
    img_variance_s0 = img_variance0+img_variance45+img_variance90+img_variance135;
    img_variance_s0 = img_variance_s0/4;
    img_variance_s1 = (img_variance0+img_variance90);
    img_variance_s2 = (img_variance45+img_variance135);
    
    mse = mean(img_variance_s0,"all");
    mean_mse = [mean_mse;mse];
    mean_mse_est = [mean_mse_est;mse/double(img_count)];


    %%

    %%
    

end

save(path_output,'mean_mse','mean_mse_est');