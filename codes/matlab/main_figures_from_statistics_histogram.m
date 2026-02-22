%% Table 2, Figure 6 noise analysis 
path_histogram = '..\..\PolarNS\histograms\histogram_statistics_merged.mat';
path_s0psnr = '..\..\PolarNS\histograms\s0psnr_merged.mat';

angles = -1:0.01:1;
angles = angles.*pi;

load(path_histogram, 'hists','hist_bins');
load(path_s0psnr,'mean_mse','mean_mse_est');

%%
log_ratio_polar_noise = 0:0.01:3;
ratio_polar_noise = 10.^log_ratio_polar_noise;
probs = pdf_aolp(angles,ratio_polar_noise');
vars = [];
for i = 1:numel(log_ratio_polar_noise)
    vars = [vars; var(angles,probs(i,:))];
end
stddevs = sqrt(vars)./2;
stddevs_degree = rad2deg(stddevs);

%%

figure('Position', [1000 200 600 600]);
subplot(3,1,1);
normalized_hist = hists.log_bias./(sum(hists.log_bias.*0.01));
plot(hist_bins.log_bias,normalized_hist,'LineWidth',1);
hold on;

normalized_hist = hists.log_bias_est./(sum(hists.log_bias_est.*0.01));
plot(hist_bins.log_bias_est,normalized_hist,'LineWidth',1);
axis([-7,0,0,1]);
% xline(-2,'LineWidth',1);
legend('Single image', 'Ours');
xticks(-6:2:0);
% xticklabels(10.^(-7:0));
xticklabels({'10^{-6}','10^{-4}','10^{-2}','1'});


subplot(3,1,2);
normalized_hist = hists.log_stddev./(sum(hists.log_stddev.*0.01));
plot(hist_bins.log_stddev,normalized_hist,'LineWidth',1);
hold on;


normalized_hist = hists.log_stddev_est./(sum(hists.log_stddev_est.*0.01));
plot(hist_bins.log_stddev_est,normalized_hist,'LineWidth',1);
axis([-4,0,0,2]);
% xline(-2,'LineWidth',1);
legend('Single image', 'Ours');
xticks(-4:0);
% xticklabels(10.^(-4:0));
xticklabels({'10^{-4}','10^{-3}','10^{-2}','10^{-1}','1'});



subplot(3,1,3);
plot(hist_bins.log_aolp_error,hists.log_aolp_error./(sum(hists.log_aolp_error.*0.01)),'LineWidth',1); 
hold on; 
plot(hist_bins.log_aolp_error,hists.log_aolp_error_est./(sum(hists.log_aolp_error_est.*0.01)),'LineWidth',1);
axis([-1,2,0,7]);
% xline(1,'LineWidth',1);
legend('Single image', 'Ours');

xticks(-1:2);
xticklabels(10.^(-1:2));

%%

fprintf("PSNR of s0 Single image: %.2f, Our dataset: %.2f\n",-10*log10(mean(mean_mse)),-10*log10(mean(mean_mse_est)));

fprintf("Bias of DoLP < 0.01  Single image: %.4f, Our dataset: %.4f\n",sum(hists.log_bias(1:600))/sum(hists.log_bias),sum(hists.log_bias_est(1:600))/sum(hists.log_bias_est));
fprintf("Bias of DoLP < 0.001 Single image: %.4f, Our dataset: %.4f\n",sum(hists.log_bias(1:500))/sum(hists.log_bias),sum(hists.log_bias_est(1:500))/sum(hists.log_bias_est));

fprintf("Standard deviation of DoLP < 0.1  Single image: %.4f, Our dataset: %.4f\n",sum(hists.log_stddev(1:400))/sum(hists.log_stddev),sum(hists.log_stddev_est(1:400))/sum(hists.log_stddev_est));
fprintf("Standard deviation of DoLP < 0.01 Single image: %.4f, Our dataset: %.4f\n",sum(hists.log_stddev(1:300))/sum(hists.log_stddev),sum(hists.log_stddev_est(1:300))/sum(hists.log_stddev_est));

fprintf("Standard deviation of AoLP < 10 degrees Single image: %.4f, Our dataset: %.4f\n",sum(hists.log_aolp_error(1:1101))/sum(hists.log_aolp_error),sum(hists.log_aolp_error_est(1:1101))/sum(hists.log_aolp_error_est));
fprintf("Standard deviation of AoLP < 5  degrees Single image: %.4f, Our dataset: %.4f\n",sum(hists.log_aolp_error(1:1071))/sum(hists.log_aolp_error),sum(hists.log_aolp_error_est(1:1071))/sum(hists.log_aolp_error_est));

