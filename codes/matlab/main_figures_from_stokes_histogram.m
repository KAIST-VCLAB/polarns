%% Supplemental material Figure 1 Stokes vector noise model
path_histogram = '..\..\PolarNS\histograms\object_stokes_direct';

angles = -1:0.01:1;
angles = angles.*pi;

load(sprintf("%s_hists_all.mat",path_histogram), 'hists','hist_bins');

%%
% hist_normalized = hists.s0_noise_s0_signal ./ sum(hists.s0_noise_s0_signal,2);
% figure; imagesc(hist_normalized);colorbar;

figure('Position', [1000 200 1200 1200]);

subplot(3,3,1);
noise_signal_sparse = reshape(hists.s0_noise_s0_signal,[5,200,5,200]); % order is quite different with numpy
noise_signal_sparse = sum(noise_signal_sparse,[1,3]);
noise_signal_sparse = squeeze(noise_signal_sparse);
hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
imagesc(flip(hist_normalized(1:100,1:100)',1));clim([0, 0.5]);
xticks(0:20:100);
xticklabels(0:0.1:0.5);
yticks(0:20:100);
yticklabels(0.00005:-0.00001:0);

% hist_normalized = hists.s1_noise_s0_signal ./ sum(hists.s1_noise_s0_signal,2);
% figure; imagesc(hist_normalized);colorbar;


subplot(3,3,2);
noise_signal_sparse = reshape(hists.s1_noise_s0_signal,[5,200,5,200]); % order is quite different with numpy
noise_signal_sparse = sum(noise_signal_sparse,[1,3]);
noise_signal_sparse = squeeze(noise_signal_sparse);
hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
imagesc(flip(hist_normalized(1:100,1:100)',1));clim([0, 0.5]);
xticks(0:20:100);
xticklabels(0:0.1:0.5);
yticks(0:20:100);
yticklabels(0.00005:-0.00001:0);

% hist_normalized = hists.s0_noise_s1_signal ./ sum(hists.s0_noise_s1_signal,2);
% figure; imagesc(hist_normalized);colorbar;


subplot(3,3,3);
noise_signal_sparse = reshape(hists.s2_noise_s0_signal,[5,200,5,200]); % order is quite different with numpy
noise_signal_sparse = sum(noise_signal_sparse,[1,3]);
noise_signal_sparse = squeeze(noise_signal_sparse);
hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
imagesc(flip(hist_normalized(1:100,1:100)',1));clim([0, 0.5]);
xticks(0:20:100);
xticklabels(0:0.1:0.5);
yticks(0:20:100);
yticklabels(0.00005:-0.00001:0);

% noise_signal_sparse = reshape(hists.s0_noise_s1_signal,[10,100,10,100]); % order is quite different with numpy
% noise_signal_sparse = sum(noise_signal_sparse,[1,3]);
% noise_signal_sparse = squeeze(noise_signal_sparse);
% hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
% figure; imagesc(hist_normalized);colorbar;
% 
% % hist_normalized = hists.s1_noise_s1_signal ./ sum(hists.s1_noise_s1_signal,2);
% % figure; imagesc(hist_normalized);colorbar;
% 
% noise_signal_sparse = reshape(hists.s1_noise_s1_signal,[10,100,10,100]); % order is quite different with numpy
% noise_signal_sparse = sum(noise_signal_sparse,[1,3]);
% noise_signal_sparse = squeeze(noise_signal_sparse);
% hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
% figure; imagesc(hist_normalized);colorbar;
% 
% noise_signal_sparse = reshape(hists.s2_noise_s1_signal,[10,100,10,100]); % order is quite different with numpy
% noise_signal_sparse = sum(noise_signal_sparse,[1,3]);
% noise_signal_sparse = squeeze(noise_signal_sparse);
% hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
% figure; imagesc(hist_normalized);colorbar;
% 
% noise_signal_sparse = reshape(hists.s0_noise_s2_signal,[10,100,10,100]); % order is quite different with numpy
% noise_signal_sparse = sum(noise_signal_sparse,[1,3]);
% noise_signal_sparse = squeeze(noise_signal_sparse);
% hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
% figure; imagesc(hist_normalized);colorbar;
% 
% noise_signal_sparse = reshape(hists.s1_noise_s2_signal,[10,100,10,100]); % order is quite different with numpy
% noise_signal_sparse = sum(noise_signal_sparse,[1,3]);
% noise_signal_sparse = squeeze(noise_signal_sparse);
% hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
% figure; imagesc(hist_normalized);colorbar;
% 
% noise_signal_sparse = reshape(hists.s2_noise_s2_signal,[10,100,10,100]); % order is quite different with numpy
% noise_signal_sparse = sum(noise_signal_sparse,[1,3]);
% noise_signal_sparse = squeeze(noise_signal_sparse);
% hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
% figure; imagesc(hist_normalized);colorbar;


subplot(3,3,4);
% s0_noise_sparse = hists.s0_noise_s1_signal;
noise_signal_sparse = reshape(hists.s0_noise_s1_signal,[1000,5,200]);
noise_signal_sparse = sum(noise_signal_sparse,[2]);
noise_signal_sparse = squeeze(noise_signal_sparse);
hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
imagesc(flip(hist_normalized(451:550,1:100)',1));clim([0, 0.5]);
xticks(1:49:100); % 0:50:100 is right, but for visualization
xticklabels(-0.1:0.1:0.1);
yticks(0:20:100);
yticklabels(0.00005:-0.00001:0);


subplot(3,3,5);
noise_signal_sparse = reshape(hists.s1_noise_s1_signal,[1000,5,200]);
noise_signal_sparse = sum(noise_signal_sparse,[2]);
noise_signal_sparse = squeeze(noise_signal_sparse);
hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
imagesc(flip(hist_normalized(451:550,1:100)',1));clim([0, 0.5]);
xticks(1:49:100); % 0:50:100 is right, but for visualization
xticklabels(-0.1:0.1:0.1);
yticks(0:20:100);
yticklabels(0.00005:-0.00001:0);



subplot(3,3,6);
noise_signal_sparse = reshape(hists.s2_noise_s1_signal,[1000,5,200]);
noise_signal_sparse = sum(noise_signal_sparse,[2]);
noise_signal_sparse = squeeze(noise_signal_sparse);
hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
imagesc(flip(hist_normalized(451:550,1:100)',1));clim([0, 0.5]);
xticks(1:49:100); % 0:50:100 is right, but for visualization
xticklabels(-0.1:0.1:0.1);
yticks(0:20:100);
yticklabels(0.00005:-0.00001:0);



subplot(3,3,7);
noise_signal_sparse = reshape(hists.s0_noise_s2_signal,[1000,5,200]);
noise_signal_sparse = sum(noise_signal_sparse,[2]);
noise_signal_sparse = squeeze(noise_signal_sparse);
hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
imagesc(flip(hist_normalized(451:550,1:100)',1));clim([0, 0.5]);
xticks(1:49:100); % 0:50:100 is right, but for visualization
xticklabels(-0.1:0.1:0.1);
yticks(0:20:100);
yticklabels(0.00005:-0.00001:0);

subplot(3,3,8);
noise_signal_sparse = reshape(hists.s1_noise_s2_signal,[1000,5,200]);
noise_signal_sparse = sum(noise_signal_sparse,[2]);
noise_signal_sparse = squeeze(noise_signal_sparse);
hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
imagesc(flip(hist_normalized(451:550,1:100)',1));clim([0, 0.5]);
xticks(1:49:100); % 0:50:100 is right, but for visualization
xticklabels(-0.1:0.1:0.1);
yticks(0:20:100);
yticklabels(0.00005:-0.00001:0);

subplot(3,3,9);
noise_signal_sparse = reshape(hists.s2_noise_s2_signal,[1000,5,200]);
noise_signal_sparse = sum(noise_signal_sparse,[2]);
noise_signal_sparse = squeeze(noise_signal_sparse);
hist_normalized = noise_signal_sparse ./ sum(noise_signal_sparse,2);
imagesc(flip(hist_normalized(451:550,1:100)',1)); clim([0, 0.5]);
% colorbar;
xticks(1:49:100); % 0:50:100 is right, but for visualization
xticklabels(-0.1:0.1:0.1);
yticks(0:20:100);
yticklabels(0.00005:-0.00001:0);
