path = 'path for raw images';

path_histogram = '..\..\PolarNS\histograms\object';

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
    
    %%
    hists = struct;
    hist_bins = struct;
    
    
    %% s0 to stokes vector noise
    hist_bins.step_s0_noise = 0.001;
    hists.s0_noise = gen_2d_histogram(img_mean_s0,img_variance_stokes*10000,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1);
    hist_bins.s0_noise = hist_bins.step_s0_noise/2:hist_bins.step_s0_noise:1;

    %%
    ratio_s0_stddev = img_mean_s0./ sqrt(img_variance_stokes);
    ratio_polar_stddev = sqrt(img_mean_polar_sq./ img_variance_stokes);

    
    hist_bins.log_polar_stddev = -4:0.01:3;
    hist_bins.log_s0 = -1:0.01:2;
    hist_bins.log_dolp = -3:0.01:0;
    hist_bins.step_log_polar_stddev = 0.01;
    hist_bins.step_log_s0 = 0.01;
    hist_bins.step_log_dolp = 0.01;
    log_ratio_polar_stddev = log10(ratio_polar_stddev);
    log_ratio_s0_stddev = log10(ratio_s0_stddev);
    log_dolp = log10(img_mean_dolp);
    hist_mask = log_ratio_polar_stddev >= -4.005 & log_ratio_polar_stddev < 3.005;
    hist_mask = hist_mask & log_ratio_s0_stddev >= -1.005 & log_ratio_s0_stddev < 2.005;
    hist_mask = hist_mask & log_dolp >= -3.005 & log_dolp < 0.005;
    masked_log_ratio_polar_stddev = log_ratio_polar_stddev(hist_mask);
    masked_log_ratio_s0_stddev = log_ratio_s0_stddev(hist_mask);
    masked_log_dolp = log_dolp(hist_mask);
    hist_idx_polar = (masked_log_ratio_polar_stddev + 4)*100;
    hist_idx_polar = round(hist_idx_polar)+1;
    hist_idx_s0 = (masked_log_ratio_s0_stddev + 1)*100;
    hist_idx_s0 = round(hist_idx_s0)+1;
    hist_idx_dolp = (masked_log_dolp + 3)*100;
    hist_idx_dolp = round(hist_idx_dolp)+1;
    
    hist_bins.angle = -180:180;
    hist_bins.step_angle = 1;
    hist_bins.step_ratio_polar_stddev = 0.01;
    hist_bins.step_obs_dolp = 0.001;
    hist_bins.ratio_polar_stddev = 0+hist_bins.step_ratio_polar_stddev/2:hist_bins.step_ratio_polar_stddev:20-hist_bins.step_ratio_polar_stddev/2;
    hist_bins.obs_dolp = 0+hist_bins.step_obs_dolp/2:hist_bins.step_obs_dolp:1.5-hist_bins.step_obs_dolp/2;
    hists.angle = zeros(numel(hist_bins.log_polar_stddev),numel(hist_bins.angle));
    hists.s0_ratio = zeros(numel(hist_bins.log_polar_stddev),numel(hist_bins.ratio_polar_stddev));
    hists.dolp = zeros(numel(hist_bins.log_s0),numel(hist_bins.log_dolp),numel(hist_bins.obs_dolp));
    img_mean_aolp = img_mean_aolp(hist_mask);
    img_mean_dolp = img_mean_dolp(hist_mask);
    img_variance_s0_masked = img_variance_stokes(hist_mask);
    
    for i = 1:img_count
        img = double(imread(fullfile(path_now,sprintf('img_gt_%04d.png',i-1))))./65536;
        [img0,img45,img90,img135] = raw2polar(img);
        img_s0 = img0+img45+img90+img135;
        img_s0 = img_s0/2;
        img_s1 = img0-img90;
        img_s2 = img45-img135;
        img_polar_sq = img_s1.^2 + img_s2.^2;
        img_aolp = atan2(img_s2,img_s1);
        img_dolp = sqrt(img_polar_sq)./img_s0;
        
        
        % dolp and aolp
        img_polar_sq = img_polar_sq(hist_mask);
        img_aolp = img_aolp(hist_mask);
        img_dolp = img_dolp(hist_mask);
        
        ratio_polar_stddev_img = sqrt(img_polar_sq./ img_variance_s0_masked);
        
        img_aolp_diff = rad2deg(diff_angles(img_aolp,img_mean_aolp));
        
        
        ratio_polar_stddev_idx = round((ratio_polar_stddev_img+hist_bins.step_ratio_polar_stddev/2)/hist_bins.step_ratio_polar_stddev);
        img_aolp_diff_idx = round(img_aolp_diff + 180)+1;
        img_dolp_idx = round((img_dolp+hist_bins.step_obs_dolp/2)/hist_bins.step_obs_dolp);
        img_aolp_diff_idx(img_aolp_diff_idx>numel(hist_bins.angle)) = numel(hist_bins.angle);
        ratio_polar_stddev_idx(ratio_polar_stddev_idx>numel(hist_bins.ratio_polar_stddev)) = numel(hist_bins.ratio_polar_stddev);
        img_dolp_idx(img_dolp_idx>numel(hist_bins.obs_dolp)) = numel(hist_bins.obs_dolp);
        
        for j = 1:numel(hist_idx_polar)
            hists.angle(hist_idx_polar(j),img_aolp_diff_idx(j)) = hists.angle(hist_idx_polar(j),img_aolp_diff_idx(j))+1;
            hists.s0_ratio(hist_idx_polar(j),ratio_polar_stddev_idx(j)) = hists.s0_ratio(hist_idx_polar(j),ratio_polar_stddev_idx(j))+1;
            hists.dolp(hist_idx_s0(j),hist_idx_dolp(j),img_dolp_idx(j)) = hists.dolp(hist_idx_s0(j),hist_idx_dolp(j),img_dolp_idx(j))+1;
        end
        
    end

    %%
    parsave(fullfile(path_histogram,sprintf("%s.mat",dir_imgs(dir_iter).name)), hists,hist_bins)
    
end

