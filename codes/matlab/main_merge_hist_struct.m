path_histogram1 = '..\..\PolarNS\histogram\histogram_statistics_dolp_hists_all.mat';
path_histogram2 = '..\..\PolarNS\histogram\histogram_statistics_aolp_hists_all.mat';
path_output = '..\..\PolarNS\histogram\histogram_statistics_merged.mat';

load(path_histogram1);
add = load(path_histogram2);
hists_add = add.hists;
fields = fieldnames(hists_add);
for idx_name = 1:numel(fieldnames(hists_add))
    hists.(fields{idx_name}) = hists_add.(fields{idx_name});
end

hist_bins_add = add.hist_bins;
fields = fieldnames(hist_bins_add);
for idx_name = 1:numel(fieldnames(hist_bins_add))
    hist_bins.(fields{idx_name}) = hist_bins_add.(fields{idx_name});
end

save(path_output, 'hists','hist_bins');