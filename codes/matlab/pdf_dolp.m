function probability = pdf_dolp(out_dolp,dolp,noise_s0_ratio)

dist = makedist('Rician','s',dolp,'sigma',noise_s0_ratio);
probability = pdf(dist,out_dolp);

end

