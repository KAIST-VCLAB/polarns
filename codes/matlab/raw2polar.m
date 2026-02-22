function [img0,img45,img90,img135] = raw2polar(img)
img90 = img(1:2:end,1:2:end);
img45 = img(1:2:end,2:2:end);
img0 = img(2:2:end,2:2:end);
img135 = img(2:2:end,1:2:end);
end

