function h = gen_1d_histogram(data,min,step,max)
    h = zeros(numel(min:step:max),1);
    idx = round((data-min)/step)+1;
    mask = idx>=1 & idx<=numel(h) & ~isnan(idx);
    idx = idx(mask);
    for i =1:numel(idx)
        h(idx(i)) = h(idx(i)) +1;
    end
end

