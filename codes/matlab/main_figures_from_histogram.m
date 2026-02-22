%% Figure 4 noise model validation 
path_histogram = '..\..\PolarNS\histograms\object';

angles = -1:0.01:1;
angles = angles.*pi;

load(sprintf("%s_hists_all.mat",path_histogram), 'hists','hist_bins');


%% real dolp

data_step = 10;

list_idx_s0ratio = [171 201 231];
list_s0ratio = [-0.7 -1 -1.3];
list_idx_dolp = [171 201 231];
list_dolp = [-1.3 -1 -0.7];




figure('Position', [1000 1000 1200 300]);

for j = 1:numel(list_idx_dolp)
    subplot(1,3,j)
    hold on;
    box on;
    for i = 1:numel(list_idx_s0ratio)
        dolp_sparse = reshape(hists.dolp(list_idx_s0ratio(i),list_idx_dolp(j),:),data_step,[]);
        dolp_sparse = squeeze(sum(dolp_sparse,1));
        dolp_sparse = dolp_sparse./sum(dolp_sparse)./(hist_bins.step_obs_dolp*data_step);
        plot(hist_bins.obs_dolp(floor(data_step/2):data_step:end),dolp_sparse,'LineWidth',0.5);
        
        fun = @(x) pdf_dolp(x,10.^list_dolp(j),10.^list_s0ratio(i));
        probs = fun(hist_bins.obs_dolp(floor(data_step/2):data_step:end));
        plot(hist_bins.obs_dolp(floor(data_step/2):data_step:end), probs,'--k','LineWidth',0.5,'HandleVisibility','off');
        % plot(hist_bins.obs_dolp(floor(data_step/2):data_step:end), probs,'LineWidth',0.5,'Color','k',LineStyle='--');
        axis([0 0.8 0 10]);
        % fontsize(16,"points")
        
    end
    xline(10.^list_dolp(j),'LineWidth',0.5,'HandleVisibility','off');
end
legend('SNR 5','SNR 10','SNR 20');


fontname('Times New Roman');
fontsize(16,"points");



%% aolp


list_idx_polar_ratio = [371 401 431];
list_polar_ratio = [-0.3 0 0.3];

figure('Position', [1000 1000 400 300]);
box on;
hold on;
for i = 1:numel(list_idx_polar_ratio)

    hist_normalized = hists.angle(list_idx_polar_ratio(i),:)./sum(uint32(hists.angle(list_idx_polar_ratio(i),:)));
    plot(hist_bins.angle/2,hist_normalized,'LineWidth',0.5);
    fun = @(x) pdf_aolp(x,10.^list_polar_ratio(i));
    probs = fun(deg2rad(hist_bins.angle));
    
    plot(hist_bins.angle/2, probs*pi./180,'--k','LineWidth',0.5,'HandleVisibility','off');
    
    axis([-89 89 0 0.015]);
    % fontsize(16,"points")
    
end
legend('SNR 0.5','SNR 1','SNR 2');
fontname('Times New Roman');
fontsize(16,"points");
