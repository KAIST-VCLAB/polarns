function h = gen_2d_histogram(data1,data2,min1,step1,max1,min2,step2,max2)
    h = zeros(numel(min1:step1:max1),numel(min2:step2:max2));
    idx1 = round((data1-min1)/step1)+1;
    idx2 = round((data2-min2)/step2)+1;
    mask = idx1>=1 & idx1<=size(h,1) & idx2>=1 & idx2<=size(h,2);
    idx1 = idx1(mask);
    idx2 = idx2(mask);
    for i =1:numel(idx1)
        h(idx1(i),idx2(i)) = h(idx1(i),idx2(i)) +1;
    end
end

