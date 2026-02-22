path = 'path for raw images';

path_histogram = '..\..\PolarNS\histogram\object_statistics_aolp';

dir_imgs = dir(path);
angles = -1:0.01:1;
angles = angles.*pi;


if ~isdir(path_histogram)
    mkdir(path_histogram)
end
dir_histogram = dir(path_histogram);
dir_histogram = struct2cell(dir_histogram);
dir_histogram = dir_histogram(1,:)';

for dir_iter = 3:numel(dir_imgs)
    if ~dir_imgs(dir_iter).isdir
        continue
    end
    
    if ~isempty(cell2mat(strfind(dir_histogram,dir_imgs(dir_iter).name)))
        continue
    end
    
%     tic;
    path_now = fullfile(path,dir_imgs(dir_iter).name);
    if ~isfile(fullfile(path_now,'img_stat.mat'))
        continue
    end
    
    
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
    img_variance_stokes = img_variance0+img_variance45+img_variance90+img_variance135;
    img_variance_stokes = img_variance_stokes/2;
    img_mean_polar_sq = img_mean_s1.^2 + img_mean_s2.^2;
    img_mean_aolp = atan2(img_mean_s2,img_mean_s1);
    img_variance_stokes_est = img_variance_stokes./double(img_count);

    %%
    hists = struct;
    hist_bins = struct;
    
    ratio_polar_stddev = sqrt(img_mean_polar_sq./ img_variance_stokes);
    ratio_polar_stddev_est = sqrt(img_mean_polar_sq./ img_variance_stokes_est);

    %% aolp
    aolp_stddev = zeros(size(img_mean_aolp));
    aolp_stddev_est = zeros(size(img_mean_aolp));
    for i = 1:size(img_mean_aolp,1)
    % for i = 551
        parfor j = 1:size(img_mean_aolp,2)
            if ~isnan(ratio_polar_stddev(i,j)) 
                probs = pdf_aolp(angles,ratio_polar_stddev(i,j));
                probs(probs<0) = 0;
                aolp_stddev(i,j) = rad2deg(std(angles,probs)./2);
            end
            if ~isnan(ratio_polar_stddev_est(i,j)) 
                probs = pdf_aolp(angles,ratio_polar_stddev_est(i,j));
                probs(probs<0) = 0;
                aolp_stddev_est(i,j) = rad2deg(std(angles,probs)./2);
            end
        end
    end
    %% aolp histogram
    % min_aolp_error = 0.05;
    % max_aolp_error = 100;
    % hist_bins.aolp_error = min_aolp_error:0.1:max_aolp_error;
    % 
    % hists.aolp_error = gen_1d_histogram(aolp_stddev,min_aolp_error,0.1,max_aolp_error);
    % hists.aolp_error_est = gen_1d_histogram(aolp_stddev_est,min_aolp_error,0.1,max_aolp_error);

    %%
    
    min_aolp_error = -10;
    max_aolp_error = 2;
    hist_bins.log_aolp_error = min_aolp_error:0.01:max_aolp_error;
    
    hists.log_aolp_error = gen_1d_histogram(log10(aolp_stddev),min_aolp_error,0.01,max_aolp_error);
    hists.log_aolp_error_est = gen_1d_histogram(log10(aolp_stddev_est),min_aolp_error,0.01,max_aolp_error);
    %%
%     save(fullfile(path_now,'histogram_v3.mat'),'hists','hist_bins');
    parsave(fullfile(path_histogram,sprintf("%s.mat",dir_imgs(dir_iter).name)), hists,hist_bins)
%     elapsed_time = toc;
%     fprintf("elapsed time:%02d:%02d:%02.2f\n",fix(elapsed_time/3600),mod(fix(elapsed_time/60),60),mod(elapsed_time,60));
    
end



% figure;
% plot(hist_bins.log_polar_stddev,hists.log_polar_stddev);
% hold on;
% 
% plot(hist_bins.log_polar_stddev_est,hists.log_polar_stddev_est);
% 
% 
% 
% 
% figure;
% plot(hist_bins.log_s0_stddev,hists.log_s0_stddev);
% hold on;
% 
% plot(hist_bins.log_s0_stddev_est,hists.log_s0_stddev_est);