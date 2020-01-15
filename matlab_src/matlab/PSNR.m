function outPSNR = PSNR(Distortion)
    outPSNR = 10*log10(255^2./Distortion);
end

