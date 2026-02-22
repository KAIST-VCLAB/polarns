path_histogram = '..\..\PolarNS\histograms\object';

angles = -1:0.01:1;
angles = angles.*pi;


dir_histogram = dir(path_histogram);

for dir_iter = 3:numel(dir_histogram)
    if dir_iter == 3
        load(fullfile(path_histogram, dir_histogram(dir_iter).name));
        fields = fieldnames(hists);
    else
        hists_add = load(fullfile(path_histogram, dir_histogram(dir_iter).name),'hists');
        hists_add = hists_add.hists;
        for idx_name = 1:numel(fieldnames(hists))
            hists.(fields{idx_name}) = hists.(fields{idx_name}) + hists_add.(fields{idx_name});
        end


    end
    % print(hist_bins)

    %% s0 noise
    % hist_normalized = hists.s0_noise ./ sum(hists.s0_noise,1);
    % figure; imagesc(hist_normalized);
    % 
    % s0_noise_sparse = reshape(hists.s0_noise,[100,10,100,10]);
    % s0_noise_sparse = sum(s0_noise_sparse,[2,4]);
    % s0_noise_sparse = squeeze(s0_noise_sparse);
    % hist_normalized = s0_noise_sparse ./ sum(s0_noise_sparse,1);
    % figure; imagesc(hist_normalized);
    %%
    
end

save(sprintf("%s_hists_all.mat",path_histogram), 'hists','hist_bins',"-v7.3");