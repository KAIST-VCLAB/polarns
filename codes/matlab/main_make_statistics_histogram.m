path = 'path for raw images';

path_histogram = '..\..\PolarNS\histogram\object_statistics_dolp';

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
    img_mean_dolp = sqrt(img_mean_polar_sq)./img_mean_s0;
    img_variance_stokes_est = img_variance_stokes./double(img_count);

    %%
    hists = struct;
    hist_bins = struct;
    
    ratio_s0_stddev = img_mean_s0./ sqrt(img_variance_stokes);
    ratio_polar_stddev = sqrt(img_mean_polar_sq./ img_variance_stokes);
    ratio_s0_stddev_est = img_mean_s0./ sqrt(img_variance_stokes_est);
    ratio_polar_stddev_est = sqrt(img_mean_polar_sq./ img_variance_stokes_est);
    % figure;imagesc(ratio_s0_stddev); colorbar;
    % figure;imagesc(ratio_polar_stddev); colorbar;
    % hist_polar_stddev = histogram(log10(ratio_polar_stddev));
    
    min_log_polar_stddev = -3;
    max_log_polar_stddev = 2;
    min_log_s0_stddev = -1;
    max_log_s0_stddev = 3;
    min_log_polar_stddev_est = -1;
    max_log_polar_stddev_est = 4;
    min_log_s0_stddev_est = 1;
    max_log_s0_stddev_est = 5;
    hist_bins.log_polar_stddev = min_log_polar_stddev:0.01:max_log_polar_stddev;
    hist_bins.log_s0_stddev = min_log_s0_stddev:0.01:max_log_s0_stddev;
    hist_bins.log_polar_stddev_est = min_log_polar_stddev_est:0.01:max_log_polar_stddev_est;
    hist_bins.log_s0_stddev_est = min_log_s0_stddev_est:0.01:max_log_s0_stddev_est;

    log_ratio_polar_stddev = log10(ratio_polar_stddev);
    log_ratio_s0_stddev = log10(ratio_s0_stddev);
    log_ratio_polar_stddev_est = log10(ratio_polar_stddev_est);
    log_ratio_s0_stddev_est = log10(ratio_s0_stddev_est);
    log_dolp = log10(img_mean_dolp);

    
    hists.log_polar_stddev = gen_1d_histogram(log_ratio_polar_stddev,min_log_polar_stddev,0.01,max_log_polar_stddev);
    hists.log_s0_stddev = gen_1d_histogram(log_ratio_s0_stddev,min_log_s0_stddev,0.01,max_log_s0_stddev);
    hists.log_polar_stddev_est = gen_1d_histogram(log_ratio_polar_stddev_est,min_log_polar_stddev_est,0.01,max_log_polar_stddev_est);
    hists.log_s0_stddev_est = gen_1d_histogram(log_ratio_s0_stddev_est,min_log_s0_stddev_est,0.01,max_log_s0_stddev_est);


    %% dolp
    % pd = makedist('Rician','s',img_mean_dolp,'sigma',1./ratio_s0_stddev);
    bias = zeros(size(img_mean_dolp));
    stddev = zeros(size(img_mean_dolp));
    bias_est = zeros(size(img_mean_dolp));
    stddev_est = zeros(size(img_mean_dolp));
    for i = 1:size(img_mean_dolp,1)
        parfor j = 1:size(img_mean_dolp,2)
            if ~isinf(ratio_s0_stddev(i,j))
                pd = makedist('Rician','s',img_mean_dolp(i,j),'sigma',1./ratio_s0_stddev(i,j));
                bias(i,j) = mean(pd)-img_mean_dolp(i,j);
                stddev(i,j) = std(pd);
            end
            if ~isinf(ratio_s0_stddev_est(i,j))
                pd = makedist('Rician','s',img_mean_dolp(i,j),'sigma',1./ratio_s0_stddev_est(i,j));
                bias_est(i,j) = mean(pd)-img_mean_dolp(i,j);
                stddev_est(i,j) = std(pd);
            end
        end
    end
    %% dolp histogram
    min_log_bias = -8;
    max_log_bias = 0;
    min_log_stddev = -5;
    max_log_stddev = 0;
    min_log_bias_est = -8;
    max_log_bias_est = -0;
    min_log_stddev_est = -5;
    max_log_stddev_est = 0;
    hist_bins.log_bias = min_log_bias:0.01:max_log_bias;
    hist_bins.log_stddev = min_log_stddev:0.01:max_log_stddev;
    hist_bins.log_bias_est = min_log_bias_est:0.01:max_log_bias_est;
    hist_bins.log_stddev_est = min_log_stddev_est:0.01:max_log_stddev_est;

    log_bias = log10(bias);
    log_stddev = log10(stddev);
    log_bias_est = log10(bias_est);
    log_stddev_est = log10(stddev_est);
    
    hists.log_bias = gen_1d_histogram(log_bias,min_log_bias,0.01,max_log_bias);
    hists.log_stddev = gen_1d_histogram(log_stddev,min_log_stddev,0.01,max_log_stddev);
    hists.log_bias_est = gen_1d_histogram(log_bias_est,min_log_bias_est,0.01,max_log_bias_est);
    hists.log_stddev_est = gen_1d_histogram(log_stddev_est,min_log_stddev_est,0.01,max_log_stddev_est);
    
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