function [diff] = diff_angles(aolp1,aolp2)
diff = mod(aolp1-aolp2 + pi, 2*pi)-pi;
end

