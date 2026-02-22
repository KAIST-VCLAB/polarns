function imshow_from_raw(img,negative)
if nargin <= 1
  negative = 0;
end
if negative
    img = img*0.5+0.5;
    imshow(demosaic(im2uint8((img)),'rggb'));
else
    img(1:2:end,1:2:end) = img(1:2:end,1:2:end)*1.67;
    img(2:2:end,2:2:end) = img(2:2:end,2:2:end)*2.30;
    imshow(demosaic(im2uint8((img).^(1.0/2.2)),'rggb'));
end
end

