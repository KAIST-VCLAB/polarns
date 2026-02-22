path = 'path for raw images';

path_histogram = '..\..\PolarNS\histogram\object_stokes_direct';

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

    loaded = load(fullfile(path_now,'img_stat_stokes.mat'));
    img_mean_s0_direct = loaded.img_mean_s0 ./ (2.^12);
    img_variance_s0_direct = loaded.img_variance_s0 ./ (2.^24);
    img_mean_s1_direct = loaded.img_mean_s1 ./ (2.^12);
    img_variance_s1_direct = loaded.img_variance_s1 ./ (2.^24);
    img_mean_s2_direct = loaded.img_mean_s2 ./ (2.^12);
    img_variance_s2_direct = loaded.img_variance_s2 ./ (2.^24);

    % figure;imshow(img_mean_s0.^(1.0./2.2));
    % figure;imshow(img_mean_s0_direct.^(1.0./2.2));
    % figure;imshow(abs(img_mean_s0-img_mean_s0_direct).^(1.0./2.2));
    % figure;imshow((img_variance_s0.*10000).^(1.0./2.2));
    % figure;imshow((img_variance_s0_direct.*10000).^(1.0./2.2));
    % figure;imshow((abs(img_variance_s0-img_variance_s0_direct).*10000).^(1.0./2.2));
    % figure;imshow((img_variance_s1.*10000).^(1.0./2.2));
    % figure;imshow((img_variance_s1_direct.*10000).^(1.0./2.2));
    % figure;imshow((abs(img_variance_s1-img_variance_s1_direct).*10000).^(1.0./2.2));
    % figure;imshow((img_variance_s2.*10000).^(1.0./2.2));
    % figure;imshow((img_variance_s2_direct.*10000).^(1.0./2.2));
    % figure;imshow((abs(img_variance_s2-img_variance_s2_direct).*10000).^(1.0./2.2));


    %%
    hists = struct;
    hist_bins = struct;
    
    %% linear fit
    
    %% s0 to stokes vector noise
    hist_bins.step_s0_noise = 0.001;
    hist_bins.step_s12_noise = 0.002;
    hist_bins.s0_noise = hist_bins.step_s0_noise/2:hist_bins.step_s0_noise:1;
    hist_bins.s12_noise = -1:hist_bins.step_s12_noise:1;
    hists.s0_noise_s0_signal = gen_2d_histogram(img_mean_s0_direct,img_variance_s0_direct*10000,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1);
    hists.s0_noise_s1_signal = gen_2d_histogram(img_mean_s1_direct,img_variance_s0_direct*10000,-1+hist_bins.step_s12_noise/2,hist_bins.step_s12_noise,1,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1);
    hists.s0_noise_s2_signal = gen_2d_histogram(img_mean_s2_direct,img_variance_s0_direct*10000,-1+hist_bins.step_s12_noise/2,hist_bins.step_s12_noise,1,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1);
    hists.s1_noise_s0_signal = gen_2d_histogram(img_mean_s0_direct,img_variance_s1_direct*10000,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1);
    hists.s1_noise_s1_signal = gen_2d_histogram(img_mean_s1_direct,img_variance_s1_direct*10000,-1+hist_bins.step_s12_noise/2,hist_bins.step_s12_noise,1,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1);
    hists.s1_noise_s2_signal = gen_2d_histogram(img_mean_s2_direct,img_variance_s1_direct*10000,-1+hist_bins.step_s12_noise/2,hist_bins.step_s12_noise,1,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1);
    hists.s2_noise_s0_signal = gen_2d_histogram(img_mean_s0_direct,img_variance_s2_direct*10000,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1);
    hists.s2_noise_s1_signal = gen_2d_histogram(img_mean_s1_direct,img_variance_s2_direct*10000,-1+hist_bins.step_s12_noise/2,hist_bins.step_s12_noise,1,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1);
    hists.s2_noise_s2_signal = gen_2d_histogram(img_mean_s2_direct,img_variance_s2_direct*10000,-1+hist_bins.step_s12_noise/2,hist_bins.step_s12_noise,1,hist_bins.step_s0_noise/2,hist_bins.step_s0_noise,1);

    %%
    
    %%
%     save(fullfile(path_now,'histogram_v3.mat'),'hists','hist_bins');
    parsave(fullfile(path_histogram,sprintf("%s.mat",dir_imgs(dir_iter).name)), hists,hist_bins)
%     elapsed_time = toc;
%     fprintf("elapsed time:%02d:%02d:%02.2f\n",fix(elapsed_time/3600),mod(fix(elapsed_time/60),60),mod(elapsed_time,60));
    
end

% %%
% 
% % hist_normalized = hists.s0_noise_s0_signal ./ sum(hists.s0_noise_s0_signal,2);
% % figure; imagesc(hist_normalized);colorbar;
% 
% s0_noise_sparse = reshape(hists.s0_noise_s0_signal,[10,100,10,100]); % order is quite different with numpy
% s0_noise_sparse = sum(s0_noise_sparse,[1,3]);
% s0_noise_sparse = squeeze(s0_noise_sparse);
% hist_normalized = s0_noise_sparse ./ sum(s0_noise_sparse,2);
% figure; imagesc(hist_normalized);colorbar;
% 
% % hist_normalized = hists.s1_noise_s0_signal ./ sum(hists.s1_noise_s0_signal,2);
% % figure; imagesc(hist_normalized);colorbar;
% 
% s0_noise_sparse = reshape(hists.s1_noise_s0_signal,[10,100,10,100]); % order is quite different with numpy
% s0_noise_sparse = sum(s0_noise_sparse,[1,3]);
% s0_noise_sparse = squeeze(s0_noise_sparse);
% hist_normalized = s0_noise_sparse ./ sum(s0_noise_sparse,2);
% figure; imagesc(hist_normalized);colorbar;
% 
% % hist_normalized = hists.s0_noise_s1_signal ./ sum(hists.s0_noise_s1_signal,2);
% % figure; imagesc(hist_normalized);colorbar;
% 
% s0_noise_sparse = reshape(hists.s0_noise_s1_signal,[10,100,10,100]); % order is quite different with numpy
% s0_noise_sparse = sum(s0_noise_sparse,[1,3]);
% s0_noise_sparse = squeeze(s0_noise_sparse);
% hist_normalized = s0_noise_sparse ./ sum(s0_noise_sparse,2);
% figure; imagesc(hist_normalized);colorbar;
% 
% % hist_normalized = hists.s1_noise_s1_signal ./ sum(hists.s1_noise_s1_signal,2);
% % figure; imagesc(hist_normalized);colorbar;
% 
% s0_noise_sparse = reshape(hists.s1_noise_s1_signal,[10,100,10,100]); % order is quite different with numpy
% s0_noise_sparse = sum(s0_noise_sparse,[1,3]);
% s0_noise_sparse = squeeze(s0_noise_sparse);
% hist_normalized = s0_noise_sparse ./ sum(s0_noise_sparse,2);
% figure; imagesc(hist_normalized);colorbar;
