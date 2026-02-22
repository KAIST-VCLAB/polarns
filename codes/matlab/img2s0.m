function s0 = img2s0(raw)

    img90 = raw(1:2:end,1:2:end);
    img45 = raw(1:2:end,2:2:end);
    img0 = raw(2:2:end,2:2:end);
    img135 = raw(2:2:end,1:2:end);
    s0 = img0+img45+img90+img135;
    s0 = s0/2;

end