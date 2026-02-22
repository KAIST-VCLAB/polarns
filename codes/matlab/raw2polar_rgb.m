function [img0,img45,img90,img135] = raw2polar_rgb(img)

img90 = im2double(demosaic(im2uint16(img(1:2:end,1:2:end)),'rggb'));
img45 = im2double(demosaic(im2uint16(img(1:2:end,2:2:end)),'rggb'));
img0 = im2double(demosaic(im2uint16(img(2:2:end,2:2:end)),'rggb'));
img135 = im2double(demosaic(im2uint16(img(2:2:end,1:2:end)),'rggb'));
end

